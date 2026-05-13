---
arxiv: '2510.25804'
authors:
- Haoran Deng
- Yingyu Lin
- Zhenghao Lin
- Xiao Liu
- Yizhou Sun
- Yi-An Ma
- Yeyun Gong
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.25804
  raw: '[[raw/papers/md/2025-beyond-length-quantifying-long-range-information-for-long-context-llm]]'
  source: https://arxiv.org/abs/2510.25804
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-beyond-length-quantifying-long-range-information-for-long-context-llm.md
raw_pdf: raw/papers/pdf/2025-beyond-length-quantifying-long-range-information-for-long-context-llm.pdf
read: false
slug: beyond-length-quantifying-long-range-information-for-long-context-llm
tags:
- type/paper
- status/stub
title: 'Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining
  Data'
type: note
updated: '2026-05-11'
year: 2025
---

# Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data

> *Haoran Deng, Yingyu Lin, Zhenghao Lin, Xiao Liu, Yizhou Sun, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Long-context language models unlock advanced capabilities in reasoning, code generation, and document summarization by leveraging dependencies across extended spans of text. However, a significant portion of readily available long-text data lacks meaningful long-distance dependencies; most spans can be predicted using only local context. Training on such data is inefficient, making careful data selection crucial. Therefore, we introduce LongFilter, a framework for curating training data tailored to long-context pretraining. LongFilter measures the information gain provided by extended context by contrasting model predictions under long-context versus short-context settings, thereby identifying samples where long-range dependencies are essential. Experiments with LLaMA-3-8B, extending its context length from 8K to 64K, show that LongFilter efficiently selects high-quality data and yields substantial improvements on benchmarks such as HELMET, LongBench, and RULER.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.25804>
- PDF: [[raw/papers/pdf/2025-beyond-length-quantifying-long-range-information-for-long-context-llm.pdf]]
- Raw markdown: [[raw/papers/md/2025-beyond-length-quantifying-long-range-information-for-long-context-llm]]
