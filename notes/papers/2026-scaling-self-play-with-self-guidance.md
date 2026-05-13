---
arxiv: '2604.20209'
authors:
- Luke Bailey
- Kaiyue Wen
- Kefan Dong
- Tatsunori Hashimoto
- Tengyu Ma
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.20209
  raw: '[[raw/papers/md/2026-scaling-self-play-with-self-guidance]]'
  source: https://arxiv.org/abs/2604.20209
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-scaling-self-play-with-self-guidance.md
raw_pdf: raw/papers/pdf/2026-scaling-self-play-with-self-guidance.pdf
read: false
slug: scaling-self-play-with-self-guidance
tags:
- type/paper
- status/stub
title: Scaling Self-Play with Self-Guidance
type: note
updated: '2026-05-11'
year: 2026
---

# Scaling Self-Play with Self-Guidance

> *Luke Bailey, Kaiyue Wen, Kefan Dong, Tatsunori Hashimoto, Tengyu Ma* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

LLM self-play algorithms are notable in that, in principle, nothing bounds their learning: a Conjecturer model creates problems for a Solver, and both improve together. However, in practice, existing LLM self-play methods do not scale well with large amounts of compute, instead hitting learning plateaus. We argue this is because over long training runs, the Conjecturer learns to hack its reward, collapsing to artificially complex problems that do not help the Solver improve. To overcome this, we introduce Self-Guided Self-Play (SGS), a self-play algorithm in which the language model itself guides the Conjecturer away from degeneracy. In SGS, the model takes on three roles: Solver, Conjecturer, and a Guide that scores synthetic problems by their relevance to unsolved target problems and how clean and natural they are, providing supervision against Conjecturer collapse. Our core hypothesis is that language models can assess whether a subproblem is useful for achieving a goal. We evaluate the scaling properties of SGS by running training for significantly longer than prior works and by fitting scaling laws to cumulative solve rate curves. Applying SGS to formal theorem proving in Lean4, we find that it surpasses the asymptotic solve rate of our strongest RL baseline in fewer than 80 rounds of self-play and enables a 7B parameter model, after 200 rounds of self-play, to solve more problems than a 671B parameter model pass@4.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.20209>
- PDF: [[raw/papers/pdf/2026-scaling-self-play-with-self-guidance.pdf]]
- Raw markdown: [[raw/papers/md/2026-scaling-self-play-with-self-guidance]]
