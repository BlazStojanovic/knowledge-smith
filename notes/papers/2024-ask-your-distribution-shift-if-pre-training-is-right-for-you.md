---
arxiv: '2403.00194'
authors:
- Benjamin Cohen-Wang
- Joshua Vendrow
- Aleksander Madry
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2403.00194
  raw: '[[raw/papers/md/2024-ask-your-distribution-shift-if-pre-training-is-right-for-you]]'
  source: https://arxiv.org/abs/2403.00194
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-ask-your-distribution-shift-if-pre-training-is-right-for-you.md
raw_pdf: raw/papers/pdf/2024-ask-your-distribution-shift-if-pre-training-is-right-for-you.pdf
read: false
slug: ask-your-distribution-shift-if-pre-training-is-right-for-you
tags:
- type/paper
- status/stub
title: Ask Your Distribution Shift if Pre-Training is Right for You
type: note
updated: '2026-05-22'
year: 2024
---

# Ask Your Distribution Shift if Pre-Training is Right for You

> *Benjamin Cohen-Wang, Joshua Vendrow, Aleksander Madry* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Pre-training is a widely used approach to develop models that are robust to distribution shifts. However, in practice, its effectiveness varies: fine-tuning a pre-trained model improves robustness significantly in some cases but not at all in others (compared to training from scratch). In this work, we seek to characterize the failure modes that pre-training can and cannot address. In particular, we focus on two possible failure modes of models under distribution shift: poor extrapolation (e.g., they cannot generalize to a different domain) and biases in the training data (e.g., they rely on spurious features). Our study suggests that, as a rule of thumb, pre-training can help mitigate poor extrapolation but not dataset biases. After providing theoretical motivation and empirical evidence for this finding, we explore two of its implications for developing robust models: (1) pre-training and interventions designed to prevent exploiting biases have complementary robustness benefits, and (2) fine-tuning on a (very) small, non-diverse but de-biased dataset can result in significantly more robust models than fine-tuning on a large and diverse but biased dataset. Code is available at https://github.com/MadryLab/pretraining-distribution-shift-robustness.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2403.00194>
- PDF: [[raw/papers/pdf/2024-ask-your-distribution-shift-if-pre-training-is-right-for-you.pdf]]
- Raw markdown: [[raw/papers/md/2024-ask-your-distribution-shift-if-pre-training-is-right-for-you]]
