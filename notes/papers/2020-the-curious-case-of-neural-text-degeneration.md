---
arxiv: '1904.09751'
authors:
- Ari Holtzman
- Jan Buys
- Li Du
- Maxwell Forbes
- Yejin Choi
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1904.09751
  raw: null
  source: https://arxiv.org/abs/1904.09751
owner: blaz
raw_pdf: raw/papers/pdf/2020-the-curious-case-of-neural-text-degeneration.pdf
read: false
slug: the-curious-case-of-neural-text-degeneration
tags:
- type/paper
- source/primary
- status/stub
- domain/inference
- domain/llm
title: The Curious Case of Neural Text Degeneration
type: note
updated: '2026-05-10'
year: 2020
---

# The Curious Case of Neural Text Degeneration

Introduces nucleus sampling (top-p) to balance diversity and plausibility in text generation.

- **Authors**: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
- **Venue**: ICLR 2020
- **arXiv**: [1904.09751](https://arxiv.org/abs/1904.09751)
- **Raw**: [[raw/papers/pdf/2020-the-curious-case-of-neural-text-degeneration]]

## Core contribution

Identifies that maximization-based decoding (beam search) leads to degenerate, repetitive text despite high model confidence. Proposes nucleus sampling (top-p sampling): truncate the vocabulary to the smallest set whose cumulative probability exceeds threshold p, then sample. Balances diversity and plausibility more effectively than temperature scaling or top-k alone.

## Connections

- Related: [[notes/papers/2018-diverse-beam-search]]
