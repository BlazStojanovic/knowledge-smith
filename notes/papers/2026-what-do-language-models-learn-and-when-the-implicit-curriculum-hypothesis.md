---
arxiv: '2604.08510'
authors:
- Emmy Liu
- Kaiser Sun
- Millicent Li
- Isabelle Lee
- Lindia Tjuatja
- Jen-tse Huang
- Graham Neubig
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.08510
  raw: '[[raw/papers/md/2026-what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis]]'
  source: https://arxiv.org/abs/2604.08510
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis.md
raw_pdf: raw/papers/pdf/2026-what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis.pdf
read: false
slug: what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis
tags:
- type/paper
- status/stub
title: What do Language Models Learn and When? The Implicit Curriculum Hypothesis
type: note
updated: '2026-05-25'
year: 2026
---

# What do Language Models Learn and When? The Implicit Curriculum Hypothesis

> *Emmy Liu, Kaiser Sun, Millicent Li, Isabelle Lee, Lindia Tjuatja, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models (LLMs) can perform remarkably complex tasks, yet the fine-grained details of how these capabilities emerge during pretraining remain poorly understood. Scaling laws on validation loss tell us how much a model improves with additional compute, but not what skills it acquires in which order. To remedy this, we propose the Implicit Curriculum Hypothesis: pretraining follows a compositional and predictable curriculum across models and data mixtures. We test this by designing a suite of simple, composable tasks spanning retrieval, morphological transformations, coreference, logical reasoning, and mathematics. Using these tasks, we track emergence points across four model families spanning sizes from 410M-13B parameters. We find that emergence orderings of when models reach fixed accuracy thresholds are strikingly consistent ($ρ= .81$ across 45 model pairs), and that composite tasks most often emerge after their component tasks. Furthermore, we find that this structure is encoded in model representations: tasks with similar function vector representations also tend to follow similar trajectories in training. By using the space of representations derived from our task set, we can effectively predict the training trajectories of simple held-out compositional tasks throughout the course of pretraining ($R^2 = .68$-$.84$ across models) without previously evaluating them. Together, these results suggest that pretraining is more structured than loss curves reveal: skills emerge in a compositional order that is consistent across models and readable from their internals.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.08510>
- PDF: [[raw/papers/pdf/2026-what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis.pdf]]
- Raw markdown: [[raw/papers/md/2026-what-do-language-models-learn-and-when-the-implicit-curriculum-hypothesis]]
