---
arxiv: '2306.07487'
authors:
- Yangruibo Ding
- Ben Steenhoek
- Kexin Pei
- Gail Kaiser
- Wei Le
- Baishakhi Ray
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2306.07487
  raw: '[[raw/papers/md/2023-traced-execution-aware-pre-training-for-source-code]]'
  source: https://arxiv.org/abs/2306.07487
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-traced-execution-aware-pre-training-for-source-code.md
raw_pdf: raw/papers/pdf/2023-traced-execution-aware-pre-training-for-source-code.pdf
read: false
slug: traced-execution-aware-pre-training-for-source-code
tags:
- type/paper
- status/stub
- pretraining
- code
- self-supervised
title: 'TRACED: Execution-aware Pre-training for Source Code'
type: note
updated: '2026-05-15'
year: 2023
---

# TRACED: Execution-aware Pre-training for Source Code

> *Yangruibo Ding, Ben Steenhoek, Kexin Pei, Gail Kaiser, Wei Le, et al.* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

Most existing pre-trained language models for source code focus on learning the static code text, typically augmented with static code structures (abstract syntax tree, dependency graphs, etc.). However, program semantics will not be fully exposed before the real execution. Without an understanding of the program execution, statically pre-trained models fail to comprehensively capture the dynamic code properties, such as the branch coverage and the runtime variable values, and they are consequently less effective at code understanding tasks, such as retrieving semantic clones and detecting software vulnerabilities.
  To close the gap between the static nature of language models and the dynamic characteristics of programs, we introduce TRACED, an execution-aware pre-training strategy for source code. Specifically, we pre-train code language models with a combination of source code, executable inputs, and corresponding execution traces. Our goal is to teach code models the complicated execution logic during the pre-training, enabling the model to statically estimate the dynamic code properties without repeatedly executing code during task-specific fine-tuning.
  To illustrate the effectiveness of our proposed approach, we fine-tune and evaluate TRACED on three downstream tasks: static execution estimation, clone retrieval, and vulnerability detection. The empirical results show that TRACED relatively improves the statically pre-trained code models by 12.4% for complete execution path prediction and by 25.2% for runtime variable value predictions. TRACED also significantly outperforms statically pre-trained models in clone retrieval and vulnerability detection across four public benchmarks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2306.07487>
- PDF: [[raw/papers/pdf/2023-traced-execution-aware-pre-training-for-source-code.pdf]]
- Raw markdown: [[raw/papers/md/2023-traced-execution-aware-pre-training-for-source-code]]
