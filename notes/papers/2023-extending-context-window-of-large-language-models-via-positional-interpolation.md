---
arxiv: '2306.15595'
authors:
- Shouyuan Chen
- Sherman Wong
- Liangjian Chen
- Yuandong Tian
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2306.15595
  raw: '[[raw/papers/md/2023-extending-context-window-of-large-language-models-via-positional-interpolation]]'
  source: https://arxiv.org/abs/2306.15595
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-extending-context-window-of-large-language-models-via-positional-interpolation.md
raw_pdf: raw/papers/pdf/2023-extending-context-window-of-large-language-models-via-positional-interpolation.pdf
read: false
slug: extending-context-window-of-large-language-models-via-positional-interpolation
tags:
- type/paper
- status/stub
title: Extending Context Window of Large Language Models via Positional Interpolation
type: note
updated: '2026-05-11'
year: 2023
---

# Extending Context Window of Large Language Models via Positional Interpolation

> *Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

We present Position Interpolation (PI) that extends the context window sizes of RoPE-based pretrained LLMs such as LLaMA models to up to 32768 with minimal fine-tuning (within 1000 steps), while demonstrating strong empirical results on various tasks that require long context, including passkey retrieval, language modeling, and long document summarization from LLaMA 7B to 65B. Meanwhile, the extended model by Position Interpolation preserve quality relatively well on tasks within its original context window. To achieve this goal, Position Interpolation linearly down-scales the input position indices to match the original context window size, rather than extrapolating beyond the trained context length which may lead to catastrophically high attention scores that completely ruin the self-attention mechanism. Our theoretical study shows that the upper bound of interpolation is at least $\sim 600 \times$ smaller than that of extrapolation, further demonstrating its stability. Models extended via Position Interpolation retain its original architecture and can reuse most pre-existing optimization and infrastructure.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2306.15595>
- PDF: [[raw/papers/pdf/2023-extending-context-window-of-large-language-models-via-positional-interpolation.pdf]]
- Raw markdown: [[raw/papers/md/2023-extending-context-window-of-large-language-models-via-positional-interpolation]]
