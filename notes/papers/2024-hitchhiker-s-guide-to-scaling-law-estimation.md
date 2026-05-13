---
arxiv: '2410.11840'
authors:
- Leshem Choshen
- Yang Zhang
- Jacob Andreas
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2410.11840
  raw: '[[raw/papers/md/2024-hitchhiker-s-guide-to-scaling-law-estimation]]'
  source: https://arxiv.org/abs/2410.11840
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-hitchhiker-s-guide-to-scaling-law-estimation.md
raw_pdf: raw/papers/pdf/2024-hitchhiker-s-guide-to-scaling-law-estimation.pdf
read: false
slug: hitchhiker-s-guide-to-scaling-law-estimation
tags:
- type/paper
- status/stub
title: A Hitchhiker's Guide to Scaling Law Estimation
type: note
updated: '2026-05-11'
year: 2024
---

# A Hitchhiker's Guide to Scaling Law Estimation

> *Leshem Choshen, Yang Zhang, Jacob Andreas* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Scaling laws predict the loss of a target machine learning model by extrapolating from easier-to-train models with fewer parameters or smaller training sets. This provides an efficient way for practitioners and researchers alike to compare pretraining decisions involving optimizers, datasets, and model architectures. Despite the widespread use of scaling laws to model the dynamics of language model training, there has been little work on understanding how to best estimate and interpret them. We collect (and release) a large-scale dataset containing losses and downstream evaluations for 485 previously published pretrained models. We use these to estimate more than 1000 scaling laws, then derive a set of best practices for estimating scaling laws in new model families. We find that fitting scaling laws to intermediate checkpoints of training runs (and not just their final losses) substantially improves accuracy, and that -- all else equal -- estimates of performance are generally most accurate when derived from other models of similar sizes. However, because there is a significant degree of variability across model seeds, training multiple small models is sometimes more useful than training a single large one. Moreover, while different model families differ scaling behavior, they are often similar enough that a target model's behavior can be predicted from a single model with the same architecture, along with scaling parameter estimates derived from other model families.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2410.11840>
- PDF: [[raw/papers/pdf/2024-hitchhiker-s-guide-to-scaling-law-estimation.pdf]]
- Raw markdown: [[raw/papers/md/2024-hitchhiker-s-guide-to-scaling-law-estimation]]
