---
aliases:
- MAD
- Alemohammad 2023
- Self-Consuming Generative Models
arxiv: '2307.01850'
authors:
- Sina Alemohammad
- Josue Casco-Rodriguez
- Lorenzo Luzi
- Ahmed Imtiaz Humayun
- Hossein Babaei
- Daniel LeJeune
- Ali Siahkoohi
- Richard G. Baraniuk
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2307.01850
  raw: https://arxiv.org/pdf/2307.01850
  source: https://arxiv.org/abs/2307.01850
owner: blaz
read: false
slug: self-consuming-generative-models-go-mad
tags:
- type/paper
- source/primary
- domain/synth-data
- status/stub
title: Self-Consuming Generative Models Go MAD (Alemohammad et al. 2023)
type: note
updated: '2026-05-10'
year: 2023
---

# Self-Consuming Generative Models Go MAD (Alemohammad et al. 2023)

## Citation

- arXiv: [2307.01850](https://arxiv.org/abs/2307.01850)
- Authors: Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, Richard G. Baraniuk.
- Year / venue: arXiv 2023.

## Core claim (stub)

Formalises **Model Autophagy Disorder (MAD)** — the failure mode where a generative model trained on its own outputs (or outputs of similar models) loses both quality and diversity over generations. Distinguishes "fully synthetic loops" (collapse is severe) from "fresh-real-data loops" (collapse is bounded). Companion result to Curse-of-Recursion ([[notes/papers/2023-the-curse-of-recursion-training-on-generated-data-makes-models-forget]]) but with a sharper distributional framing of what fails.

## Why it's load-bearing

One of the two foundational results in the iteration-dynamics literature ([[concepts/iteration-dynamics]]). The collapse-vs-drift distinction in our taxonomy traces directly to MAD's framing.

## Status

Stub. Full read deferred. The vault references this paper in multiple places (concepts/diversity.md, concepts/iteration-dynamics.md, maps/evaluation/distributional-level.md) without a paper note — this stub closes that loop.
