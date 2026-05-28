---
arxiv: '2605.26099'
authors:
- Sangyun Lee
- Sean McLeish
- Tom Goldstein
- Giulia Fanti
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.26099
  raw: '[[raw/papers/md/2026-language-models-need-sleep]]'
  source: https://arxiv.org/abs/2605.26099
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-language-models-need-sleep.md
raw_pdf: raw/papers/pdf/2026-language-models-need-sleep.pdf
read: false
slug: language-models-need-sleep
tags:
- type/paper
- status/stub
title: Language Models Need Sleep
type: note
updated: '2026-05-28'
year: 2026
---

# Language Models Need Sleep

> *Sangyun Lee, Sean McLeish, Tom Goldstein, Giulia Fanti* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.26099>
- PDF: [[raw/papers/pdf/2026-language-models-need-sleep.pdf]]
- Raw markdown: [[raw/papers/md/2026-language-models-need-sleep]]
