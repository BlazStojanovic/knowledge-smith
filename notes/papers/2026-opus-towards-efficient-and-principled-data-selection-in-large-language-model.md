---
arxiv: '2602.05400'
authors:
- Shaobo Wang
- Xuan Ouyang
- Tianyi Xu
- Yuzheng Hu
- Jialin Liu
- Guo Chen
- Tianyu Zhang
- Junhao Zheng
- Kexin Yang
- Xingzhang Ren
- Dayiheng Liu
- Linfeng Zhang
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.05400
  raw: '[[raw/papers/md/2026-opus-towards-efficient-and-principled-data-selection-in-large-language-model]]'
  source: https://arxiv.org/abs/2602.05400
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-opus-towards-efficient-and-principled-data-selection-in-large-language-model.md
raw_pdf: raw/papers/pdf/2026-opus-towards-efficient-and-principled-data-selection-in-large-language-model.pdf
read: false
slug: opus-towards-efficient-and-principled-data-selection-in-large-language-model
tags:
- type/paper
- status/stub
title: 'OPUS: Towards Efficient and Principled Data Selection in Large Language Model
  Pre-training in Every Iteration'
type: note
updated: '2026-05-25'
year: 2026
---

# OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration

> *Shaobo Wang, Xuan Ouyang, Tianyi Xu, Yuzheng Hu, Jialin Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall, pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose OPUS (Optimizer-induced Projected Utility Selection), a dynamic data selection framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7\% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. In pre-training of GPT-2 Large/XL on FineWeb and FineWeb-Edu with 30B tokens, OPUS outperforms industrial-level baselines and even full 200B-token training. Moreover, when combined with industrial-level static filters, OPUS further improves pre-training efficiency, even with lower-quality data. Furthermore, in continued pre-training of Qwen3-8B-Base on SciencePedia, OPUS achieves superior performance using only 0.5B tokens compared to full training with 3B tokens, demonstrating significant data efficiency gains in specialized domains.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.05400>
- PDF: [[raw/papers/pdf/2026-opus-towards-efficient-and-principled-data-selection-in-large-language-model.pdf]]
- Raw markdown: [[raw/papers/md/2026-opus-towards-efficient-and-principled-data-selection-in-large-language-model]]
