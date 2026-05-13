---
arxiv: '2207.14255'
authors:
- Mohammad Bavarian
- Heewoo Jun
- Nikolas Tezak
- John Schulman
- Christine McLeavey
- Jerry Tworek
- Mark Chen
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2207.14255
  raw: null
  source: https://arxiv.org/abs/2207.14255
owner: blaz
raw_pdf: raw/papers/pdf/2022-fim-training.pdf
read: false
slug: fim-training
tags:
- type/paper
- source/primary
- status/stub
- domain/code
- domain/training
title: Efficient Training of Language Models to Fill in the Middle
type: note
updated: '2026-05-10'
year: 2022
---

# Efficient Training of Language Models to Fill in the Middle

Demonstrates that autoregressive LMs can learn to infill by moving a span from the middle of a document to the end during training — no architecture changes needed.

- **Authors**: Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, Mark Chen
- **Venue**: arXiv 2022 (OpenAI)
- **arXiv**: [2207.14255](https://arxiv.org/abs/2207.14255)
- **Raw**: [[raw/papers/pdf/2022-fim-training]]

## Core contribution

Fill-in-the-middle (FIM) training: a simple data transformation (move a middle span to the end) teaches autoregressive models to infill. No architecture change, minimal compute overhead, does not degrade left-to-right capabilities.

## Methodological relevance

FIM evaluation requires different metrics than left-to-right generation:
- Infilling accuracy (exact match or execution-based)
- The evaluation must condition on both prefix and suffix context
- Creates a distinct evaluation paradigm: completion-in-context

## Connections

- Map: [[maps/model-evaluation/code-eval-paradigms]] — FIM as distinct task type
- Related: [[notes/papers/2021-evaluating-large-language-models-trained-on-code]], [[notes/papers/2022-alphacode]]
