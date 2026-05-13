---
arxiv: '1610.02424'
authors:
- Ashwin K. Vijayakumar
- Michael Cogswell
- Ramprasath R. Selvaraju
- Qing Sun
- Stefan Lee
- David Crandall
- Dhruv Batra
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1610.02424
  raw: null
  source: https://arxiv.org/abs/1610.02424
owner: blaz
raw_pdf: raw/papers/pdf/2018-diverse-beam-search.pdf
read: false
slug: diverse-beam-search
tags:
- type/paper
- source/primary
- status/stub
- domain/inference
- domain/llm
title: 'Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models'
type: note
updated: '2026-05-10'
year: 2018
---

# Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models

Diversity-promoting beam search that prevents redundancy in decoded outputs.

- **Authors**: Ashwin K. Vijayakumar, Michael Cogswell, Ramprasath R. Selvaraju, Qing Sun, Stefan Lee, David Crandall, Dhruv Batra
- **Venue**: AAAI 2018
- **arXiv**: [1610.02424](https://arxiv.org/abs/1610.02424)
- **Raw**: [[raw/papers/pdf/2018-diverse-beam-search]]

## Core contribution

Modifies beam search to optimize for diversity by partitioning beams into groups and adding a diversity-promoting penalty between groups. Produces a set of diverse, high-quality candidate sequences rather than near-duplicate variants of the single best sequence.

## Connections

- Related: [[notes/papers/2020-the-curious-case-of-neural-text-degeneration]]
