---
arxiv: '2601.18778'
authors:
- Shobhita Sundaram
- John Quan
- Ariel Kwiatkowski
- Kartik Ahuja
- Yann Ollivier
- Julia Kempe
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2026-teaching-models-to-teach-themselves-reasoning-at-the-edge.md
raw_pdf: raw/papers/pdf/2026-teaching-models-to-teach-themselves-reasoning-at-the-edge.pdf
read: false
slug: teaching-models-to-teach-themselves-reasoning-at-the-edge
tags:
- llm
- self-supervised
- fine-tuning
title: 'Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability'
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2601.18778
venue: null
year: 2026
---

# Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

> *Shobhita Sundaram, John Quan, Ariel Kwiatkowski…* — arXiv 2601.18778, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Can a model learn to escape its own learning plateau? Reinforcement learning methods for finetuning large reasoning models stall on datasets with low initial success rates, and thus little training signal. This work investigates whether a pretrained LLM can leverage latent knowledge to generate an automated curriculum for problems it cannot solve. The authors design SOAR, a self-improvement framework that surfaces pedagogical signals through meta-RL. A teacher copy of the model proposes synthetic problems for a student copy, and is rewarded with its improvement on a small subset of hard problems. The study on the hardest subsets of mathematical benchmarks reveals that bi-level meta-RL can unlock learning under sparse, binary rewards by sharpening a latent capacity of pretrained models to generate useful stepping stones. Grounded rewards outperform intrinsic reward schemes used in prior LLM self-play, reliably avoiding instability and diversity collapse modes. Analysis of the generated questions reveals that structural quality and well-posedness are more critical for learning progress than solution correctness. The results suggest that the ability to generate useful stepping stones does not require the preexisting ability to actually solve the hard problems, paving a principled path to escape reasoning plateaus without additional curated data.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-teaching-models-to-teach-themselves-reasoning-at-the-edge]]
- PDF: [[raw/papers/pdf/2026-teaching-models-to-teach-themselves-reasoning-at-the-edge.pdf]]
- arXiv: <https://arxiv.org/abs/2601.18778>
