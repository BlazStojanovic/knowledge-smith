---
arxiv: '2203.07814'
authors:
- Yujia Li
- David Choi
- Junyoung Chung
- Nate Kushman
- Julian Schrittwieser
- Rémi Leblond
- Tom Eccles
- et al. (DeepMind)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2203.07814
  raw: null
  source: https://arxiv.org/abs/2203.07814
owner: blaz
raw_pdf: raw/papers/pdf/2022-alphacode.pdf
read: false
slug: alphacode
tags:
- type/paper
- source/primary
- status/stub
- domain/evals
- domain/code
title: Competition-Level Code Generation with AlphaCode
type: note
updated: '2026-05-10'
year: 2022
---

# Competition-Level Code Generation with AlphaCode

First system to achieve human-competitive performance on competitive programming (Codeforces), ranking top 54.3% among 5,000+ participants.

- **Authors**: Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, et al. (DeepMind)
- **Venue**: Science 2022 (arXiv 2022)
- **arXiv**: [2203.07814](https://arxiv.org/abs/2203.07814)
- **Raw**: [[raw/papers/pdf/2022-alphacode]]

## Core contribution

Demonstrates that massive sampling + filtering can solve competition-level problems. Key innovation: generate ~1M candidate solutions per problem, then cluster and filter using test cases to select ~10 submissions.

## Methodological relevance

- **Sampling at scale**: extends pass@k paradigm to extreme k (millions), with clustering-based selection replacing random sampling
- **Competition evaluation**: uses Codeforces rating system as metric (Elo-like), not just pass@k
- **Hidden test cases**: competition format provides strong anti-contamination (test cases are not public)

## Key results

- Top 54.3% ranking on Codeforces among 5,000+ participants
- Required ~1M samples per problem with clustering + filtering

## Connections

- Map: [[maps/model-evaluation/code-eval-paradigms]] — competition-level, execution-based, massive sampling
- Concepts: [[concepts/pass-at-k-methodology]] (extreme-k regime)
- Related: [[notes/papers/2021-evaluating-large-language-models-trained-on-code]], [[notes/papers/2021-apps]]
