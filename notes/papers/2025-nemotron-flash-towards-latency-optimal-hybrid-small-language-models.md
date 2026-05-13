---
arxiv: '2511.18890'
authors:
- Yonggan Fu
- Xin Dong
- Shizhe Diao
- Matthijs Van keirsbilck
- Hanrong Ye
- Wonmin Byeon
- Yashaswi Karnati
- Lucas Liebenwein
- Hannah Zhang
- Nikolaus Binder
- Maksim Khadkevich
- Alexander Keller
- Jan Kautz
- Yingyan Celine Lin
- Pavlo Molchanov
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2511.18890
  raw: '[[raw/papers/md/2025-nemotron-flash-towards-latency-optimal-hybrid-small-language-models]]'
  source: https://arxiv.org/abs/2511.18890
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-nemotron-flash-towards-latency-optimal-hybrid-small-language-models.md
raw_pdf: raw/papers/pdf/2025-nemotron-flash-towards-latency-optimal-hybrid-small-language-models.pdf
read: false
slug: nemotron-flash-towards-latency-optimal-hybrid-small-language-models
tags:
- type/paper
- status/stub
title: 'Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models'
type: note
updated: '2026-05-11'
year: 2025
---

# Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models

> *Yonggan Fu, Xin Dong, Shizhe Diao, Matthijs Van keirsbilck, Hanrong Ye, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Efficient deployment of small language models (SLMs) is essential for numerous real-world applications with stringent latency constraints. While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups. This work aims to identify the key determinants of SLMs' real-device latency and offer generalizable principles and methodologies for SLM design and training when real-device latency is the primary consideration. Specifically, we identify two central architectural factors: depth-width ratios and operator choices. The former is crucial for small-batch-size latency, while the latter affects both latency and large-batch-size throughput. In light of this, we first study latency-optimal depth-width ratios, with the key finding that although deep-thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy-latency trade-off frontier. Next, we explore emerging efficient attention alternatives to evaluate their potential as candidate building operators. Using the identified promising operators, we construct an evolutionary search framework to automatically discover latency-optimal combinations of these operators within hybrid SLMs, thereby advancing the accuracy-latency frontier. In addition to architectural improvements, we further enhance SLM training using a weight normalization technique that enables more effective weight updates and improves final convergence. Combining these methods, we introduce a new family of hybrid SLMs, called Nemotron-Flash, which significantly advances the accuracy-efficiency frontier of state-of-the-art SLMs, e.g., achieving over +5.5% average accuracy, 1.3x/1.9x lower latency, and 18.7x/45.6x higher throughput compared to Qwen3-1.7B/0.6B, respectively.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2511.18890>
- PDF: [[raw/papers/pdf/2025-nemotron-flash-towards-latency-optimal-hybrid-small-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2025-nemotron-flash-towards-latency-optimal-hybrid-small-language-models]]
