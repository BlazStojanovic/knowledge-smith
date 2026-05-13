---
arxiv: '2604.08423'
authors:
- Tristan Thrush
- Sung Min Park
- Herman Brunborg
- Luke Bailey
- Marcel Roed
- Neil Band
- Christopher Potts
- Tatsunori Hashimoto
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.08423
  raw: '[[raw/papers/md/2026-synthetic-data-for-any-differentiable-target]]'
  source: https://arxiv.org/abs/2604.08423
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-synthetic-data-for-any-differentiable-target.md
raw_pdf: raw/papers/pdf/2026-synthetic-data-for-any-differentiable-target.pdf
read: false
slug: synthetic-data-for-any-differentiable-target
tags:
- type/paper
- status/stub
title: Synthetic Data for any Differentiable Target
type: note
updated: '2026-05-11'
year: 2026
---

# Synthetic Data for any Differentiable Target

> *Tristan Thrush, Sung Min Park, Herman Brunborg, Luke Bailey, Marcel Roed, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

What are the limits of controlling language models via synthetic training data? We develop a reinforcement learning (RL) primitive, the Dataset Policy Gradient (DPG), which can precisely optimize synthetic data generators to produce a dataset of targeted examples. When used for supervised fine-tuning (SFT) of a target model, these examples cause the target model to do well on a differentiable metric of our choice. Our approach achieves this by taking exact data attribution via higher-order gradients and using those scores as policy gradient rewards. We prove that this procedure closely approximates the true, intractable gradient for the synthetic data generator. To illustrate the potential of DPG, we show that, using only SFT on generated examples, we can cause the target model's LM head weights to (1) embed a QR code, (2) embed the pattern $\texttt{67}$, and (3) have lower $\ell^2$ norm. We additionally show that we can cause the generator to (4) rephrase inputs in a new language and (5) produce a specific UUID, even though neither of these objectives is conveyed in the generator's input prompts. These findings suggest that DPG is a powerful and flexible technique for shaping model properties using only synthetic training examples.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.08423>
- PDF: [[raw/papers/pdf/2026-synthetic-data-for-any-differentiable-target.pdf]]
- Raw markdown: [[raw/papers/md/2026-synthetic-data-for-any-differentiable-target]]
