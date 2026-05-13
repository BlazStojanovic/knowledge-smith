---
arxiv: '2603.26535'
authors:
- Zelin Tan
- Zhouliang Yu
- Bohan Lin
- Zijie Geng
- Hejia Geng
- Yudong Zhang
- Mulei Zhang
- Yang Chen
- Shuyue Hu
- Zhenfei Yin
- Chen Zhang
- Lei Bai
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.26535
  raw: '[[raw/papers/md/2026-papo-stabilizing-rubric-integration-training-via-decoupled-advantage]]'
  source: https://arxiv.org/abs/2603.26535
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-papo-stabilizing-rubric-integration-training-via-decoupled-advantage.md
raw_pdf: raw/papers/pdf/2026-papo-stabilizing-rubric-integration-training-via-decoupled-advantage.pdf
read: false
slug: papo-stabilizing-rubric-integration-training-via-decoupled-advantage
tags:
- type/paper
- status/stub
title: 'PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization'
type: note
updated: '2026-05-11'
year: 2026
---

# PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization

> *Zelin Tan, Zhouliang Yu, Bohan Lin, Zijie Geng, Hejia Geng, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We propose Process-Aware Policy Optimization (PAPO), a method that integrates process-level evaluation into Group Relative Policy Optimization (GRPO) through decoupled advantage normalization, to address two limitations of existing reward designs. Outcome reward models (ORM) evaluate only final-answer correctness, treating all correct responses identically regardless of reasoning quality, and gradually lose the advantage signal as groups become uniformly correct. Process reward models (PRM) offer richer supervision, but directly using PRM scores causes reward hacking, where models exploit verbosity to inflate scores while accuracy collapses. PAPO resolves both by composing the advantage from an outcome component Aout, derived from ORM and normalized over all responses, and a process component Aproc, derived from a rubric-based PRM and normalized exclusively among correct responses. This decoupled design ensures that Aout anchors training on correctness while Aproc differentiates reasoning quality without distorting the outcome signal. Experiments across multiple model scales and six benchmarks demonstrate that PAPO consistently outperforms ORM, reaching 51.3% vs.\ 46.3% on OlympiadBench while continuing to improve as ORM plateaus and declines.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.26535>
- PDF: [[raw/papers/pdf/2026-papo-stabilizing-rubric-integration-training-via-decoupled-advantage.pdf]]
- Raw markdown: [[raw/papers/md/2026-papo-stabilizing-rubric-integration-training-via-decoupled-advantage]]
