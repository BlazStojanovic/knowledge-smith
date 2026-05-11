---
arxiv: '2512.10858'
authors:
- Dimitri von Rütte
- Janis Fluri
- Omead Pooladzandi
- Bernhard Schölkopf
- Thomas Hofmann
- Antonio Orvieto
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.10858
  raw: '[[raw/papers/md/2025-scaling-behavior-of-discrete-diffusion-language-models]]'
  source: https://arxiv.org/abs/2512.10858
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-scaling-behavior-of-discrete-diffusion-language-models.md
raw_pdf: raw/papers/pdf/2025-scaling-behavior-of-discrete-diffusion-language-models.pdf
read: false
slug: scaling-behavior-of-discrete-diffusion-language-models
tags:
- type/paper
- diffusion
- llm
- scaling-laws
- status/stub
title: Scaling Behavior of Discrete Diffusion Language Models
type: note
updated: '2026-05-09'
venue: null
year: 2025
---

# Scaling Behavior of Discrete Diffusion Language Models

> *Dimitri von Rütte, Janis Fluri, Omead Pooladzandi…* — arXiv 2512.10858, 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Modern LLM pre-training consumes vast amounts of compute and training data, making the scaling behavior, or scaling laws, of different models a key distinguishing factor. Discrete diffusion language models (DLMs) have been proposed as an alternative to autoregressive language models (ALMs). However, their scaling behavior has not yet been fully explored, with prior work suggesting that they require more data and compute to match the performance of ALMs. We study the scaling behavior of DLMs on different noise types by smoothly interpolating between masked and uniform diffusion while paying close attention to crucial hyperparameters such as batch size and learning rate. Our experiments reveal that the scaling behavior of DLMs strongly depends on the noise type and is considerably different from ALMs. While all noise types converge to similar loss values in compute-bound scaling, we find that uniform diffusion requires more parameters and less data for compute-efficient training compared to masked diffusion, making them a promising candidate in data-bound settings. We scale our uniform diffusion model up to 10B parameters trained for 10^22 FLOPs, confirming the predicted scaling behavior and making it the largest publicly known uniform diffusion model to date.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2025-scaling-behavior-of-discrete-diffusion-language-models]]
- PDF: [[raw/papers/pdf/2025-scaling-behavior-of-discrete-diffusion-language-models.pdf]]
- arXiv: <https://arxiv.org/abs/2512.10858>
