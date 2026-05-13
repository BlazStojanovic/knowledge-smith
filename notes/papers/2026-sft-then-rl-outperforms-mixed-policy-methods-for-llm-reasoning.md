---
arxiv: '2604.23747'
authors:
- Alexis Limozin
- Eduard Durech
- Torsten Hoefler
- Imanol Schlag
- Valentina Pyatkin
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.23747
  raw: '[[raw/papers/md/2026-sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning]]'
  source: https://arxiv.org/abs/2604.23747
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning.md
raw_pdf: raw/papers/pdf/2026-sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning.pdf
read: false
slug: sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning
tags:
- type/paper
- status/stub
title: SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning
type: note
updated: '2026-05-11'
year: 2026
---

# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning

> *Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.23747>
- PDF: [[raw/papers/pdf/2026-sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning.pdf]]
- Raw markdown: [[raw/papers/md/2026-sft-then-rl-outperforms-mixed-policy-methods-for-llm-reasoning]]
