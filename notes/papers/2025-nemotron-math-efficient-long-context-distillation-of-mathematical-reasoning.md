---
arxiv: '2512.15489'
authors:
- Wei Du
- Shubham Toshniwal
- Branislav Kisacanin
- Sadegh Mahdavi
- Ivan Moshkov
- George Armstrong
- Stephen Ge
- Edgar Minasyan
- Feng Chen
- Igor Gitman
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.15489
  raw: '[[raw/papers/md/2025-nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning]]'
  source: https://arxiv.org/abs/2512.15489
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning.md
raw_pdf: raw/papers/pdf/2025-nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning.pdf
read: false
slug: nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning
tags:
- type/paper
- status/stub
- math
- synthetic-data
- distillation
- reasoning
- long-context
title: 'Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning
  from Multi-Mode Supervision'
type: note
updated: '2026-05-18'
year: 2025
---

# Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision

> *Wei Du, Shubham Toshniwal, Branislav Kisacanin, Sadegh Mahdavi, Ivan Moshkov, et al.* — arXiv 2025

## TL;DR

Large-scale math-reasoning SFT dataset distilled from `gpt-oss-120b` using its multi-mode generation: **7.5M solution traces** spanning high/medium/low reasoning modes, each produced both with and without Python tool-integrated reasoning (TIR). Sources: 85K curated AoPS competition problems + 262K community-sourced StackExchange-Math problems — structured competition tasks plus messier real-world queries. Controlled evals show Nemotron-Math beats the original OpenMathReasoning on matched AoPS problems; adding StackExchange-Math improves robustness and generalization (notably HLE-Math) without hurting competition benchmarks. To make long-context training affordable, a **sequential bucketed strategy** speeds up 128K-context fine-tuning 2–3× with negligible accuracy loss. Reported SOTA: 100% maj@16 on AIME 2024 and 2025 with Python TIR. (Summary from abstract; note unread.)

## Abstract

High-quality mathematical reasoning supervision requires diverse reasoning styles, long-form traces, and effective tool integration, capabilities that existing datasets provide only in limited form. Leveraging the multi-mode generation ability of gpt-oss-120b, we introduce Nemotron-Math, a large-scale mathematical reasoning dataset containing 7.5M solution traces across high, medium, and low reasoning modes, each available both with and without Python tool-integrated reasoning (TIR).
  The dataset integrates 85K curated AoPS problems with 262K community-sourced StackExchange-Math problems, combining structured competition tasks with diverse real-world mathematical queries. We conduct controlled evaluations to assess the dataset quality.
  Nemotron-Math consistently outperforms the original OpenMathReasoning on matched AoPS problems. Incorporating StackExchange-Math substantially improves robustness and generalization, especially on HLE-Math, while preserving accuracy on math competition benchmarks.
  To support efficient long-context training, we develop a sequential bucketed strategy that accelerates 128K context-length fine-tuning by 2--3$\times$ without significant accuracy loss. Overall, Nemotron-Math enables state-of-the-art performance, including 100\% maj@16 accuracy on AIME 2024 and 2025 with Python TIR.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.15489>
- PDF: [[raw/papers/pdf/2025-nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning.pdf]]
- Raw markdown: [[raw/papers/md/2025-nemotron-math-efficient-long-context-distillation-of-mathematical-reasoning]]
