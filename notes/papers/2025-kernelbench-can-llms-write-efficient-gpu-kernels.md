---
arxiv: '2502.10517'
authors:
- Anne Ouyang
- Simon Guo
- Simran Arora
- Alex L. Zhang
- William Hu
- Christopher Ré
- Azalia Mirhoseini
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.10517
  raw: '[[raw/papers/md/2025-kernelbench-can-llms-write-efficient-gpu-kernels]]'
  source: https://arxiv.org/abs/2502.10517
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-kernelbench-can-llms-write-efficient-gpu-kernels.md
raw_pdf: raw/papers/pdf/2025-kernelbench-can-llms-write-efficient-gpu-kernels.pdf
read: false
slug: kernelbench-can-llms-write-efficient-gpu-kernels
tags:
- type/paper
- status/stub
- benchmark
- code-generation
- evaluation
- llm
title: 'KernelBench: Can LLMs Write Efficient GPU Kernels?'
type: note
updated: '2026-05-15'
year: 2025
---

# KernelBench: Can LLMs Write Efficient GPU Kernels?

> *Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Efficient GPU kernels are crucial for building performant machine learning architectures, but writing them is a time-consuming challenge that requires significant expertise; therefore, we explore using language models (LMs) to automate kernel generation. We introduce KernelBench, an open-source framework for evaluating LMs' ability to write fast and correct kernels on a suite of 250 carefully selected PyTorch ML workloads. KernelBench represents a real-world engineering environment and making progress on the introduced benchmark directly translates to faster practical kernels. We introduce a new evaluation metric fast_p, which measures the percentage of generated kernels that are functionally correct and offer a speedup greater than an adjustable threshold p over baseline. Our experiments across various state-of-the-art models and test-time methods show that frontier reasoning models perform the best out of the box but still fall short overall, matching the PyTorch baseline in less than 20% of the cases. While we show that results can improve by leveraging execution and profiling feedback during iterative refinement, KernelBench remains a challenging benchmark, with its difficulty increasing as we raise speedup threshold p.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2502.10517>
- PDF: [[raw/papers/pdf/2025-kernelbench-can-llms-write-efficient-gpu-kernels.pdf]]
- Raw markdown: [[raw/papers/md/2025-kernelbench-can-llms-write-efficient-gpu-kernels]]
