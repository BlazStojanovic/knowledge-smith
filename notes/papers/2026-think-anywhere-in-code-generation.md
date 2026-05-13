---
arxiv: '2603.29957'
authors:
- Xue Jiang
- Tianyu Zhang
- Ge Li
- Mengyang Liu
- Taozhi Chen
- Zhenhua Xu
- Binhua Li
- Wenpin Jiao
- Zhi Jin
- Yongbin Li
- Yihong Dong
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.29957
  raw: '[[raw/papers/md/2026-think-anywhere-in-code-generation]]'
  source: https://arxiv.org/abs/2603.29957
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-think-anywhere-in-code-generation.md
raw_pdf: raw/papers/pdf/2026-think-anywhere-in-code-generation.pdf
read: false
slug: think-anywhere-in-code-generation
tags:
- type/paper
- status/stub
title: Think Anywhere in Code Generation
type: note
updated: '2026-05-11'
year: 2026
---

# Think Anywhere in Code Generation

> *Xue Jiang, Tianyu Zhang, Ge Li, Mengyang Liu, Taozhi Chen, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Recent advances in reasoning Large Language Models (LLMs) have primarily relied on upfront thinking, where reasoning occurs before final answer. However, this approach suffers from critical limitations in code generation, where upfront thinking is often insufficient as problems' full complexity only reveals itself during code implementation. Moreover, it cannot adaptively allocate reasoning effort throughout the code generation process where difficulty varies significantly. In this paper, we propose Think-Anywhere, a novel reasoning mechanism that enables LLMs to invoke thinking on-demand at any token position during code generation. We achieve Think-Anywhere by first teaching LLMs to imitate the reasoning patterns through cold-start training, then leveraging outcome-based RL rewards to drive the model's autonomous exploration of when and where to invoke reasoning. Extensive experiments on four mainstream code generation benchmarks (i.e., LeetCode, LiveCodeBench, HumanEval, and MBPP) show that Think-Anywhere achieves state-of-the-art performance over both existing reasoning methods and recent post-training approaches, while demonstrating consistent generalization across diverse LLMs. Our analysis further reveals that Think-Anywhere enables the model to adaptively invoke reasoning at high-entropy positions, providing enhanced interpretability.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.29957>
- PDF: [[raw/papers/pdf/2026-think-anywhere-in-code-generation.pdf]]
- Raw markdown: [[raw/papers/md/2026-think-anywhere-in-code-generation]]
