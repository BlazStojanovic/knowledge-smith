---
arxiv: '2605.06546'
authors:
- Bowen Peng
- Théo Gigant
- Jeffrey Quesnelle
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.06546
  raw: '[[raw/papers/md/2026-efficient-pre-training-with-token-superposition]]'
  source: https://arxiv.org/abs/2605.06546
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-efficient-pre-training-with-token-superposition.md
raw_pdf: raw/papers/pdf/2026-efficient-pre-training-with-token-superposition.pdf
read: false
slug: efficient-pre-training-with-token-superposition
tags:
- type/paper
- status/stub
- pretraining
- efficiency
- llm
title: Efficient Pre-Training with Token Superposition
type: note
updated: '2026-05-15'
year: 2026
---

# Efficient Pre-Training with Token Superposition

> *Bowen Peng, Théo Gigant, Jeffrey Quesnelle* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Pre-training of Large Language Models is often prohibitively expensive and inefficient at scale, requiring complex and invasive modifications in order to achieve high data throughput. In this work, we present Token-Superposition Training (TST), a simple drop-in method that significantly improves the data throughput per FLOPs during pre-training without modifying the parallelism, optimizer, tokenizer, data, or model architecture. TST is done in two phases: (i) A highly efficient superposition phase where we combine many contiguous tokens into one bag and train using a multi-hot cross-entropy (MCE) objective, and (ii) a recovery phase where we revert back to standard training. We extensively evaluate TST on the scale of 270M and 600M parameters and validate on 3B and a 10B A1B mixture of experts model, demonstrating that it is highly robust in different settings. Ultimately, TST consistently outperforms baseline loss and downstream evaluations, and under equal-loss settings, TST yields up to a 2.5x reduction in total pre-training time at the 10B A1B scale.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.06546>
- PDF: [[raw/papers/pdf/2026-efficient-pre-training-with-token-superposition.pdf]]
- Raw markdown: [[raw/papers/md/2026-efficient-pre-training-with-token-superposition]]
