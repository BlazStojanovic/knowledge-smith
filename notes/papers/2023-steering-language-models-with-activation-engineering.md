---
arxiv: '2308.10248'
authors:
- Alexander Matt Turner
- Lisa Thiergart
- Gavin Leech
- David Udell
- Juan J. Vazquez
- Ulisse Mini
- Monte MacDiarmid
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2308.10248
  raw: '[[raw/papers/md/2023-steering-language-models-with-activation-engineering]]'
  source: https://arxiv.org/abs/2308.10248
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-steering-language-models-with-activation-engineering.md
raw_pdf: raw/papers/pdf/2023-steering-language-models-with-activation-engineering.pdf
read: false
slug: steering-language-models-with-activation-engineering
tags:
- type/paper
- status/stub
title: Steering Language Models With Activation Engineering
type: note
updated: '2026-05-11'
year: 2023
---

# Steering Language Models With Activation Engineering

> *Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, et al.* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

Prompt engineering and finetuning aim to maximize language model performance on a given metric (like toxicity reduction). However, these methods do not fully elicit a model's capabilities. To reduce this gap, we introduce activation engineering: the inference-time modification of activations in order to control (or steer) model outputs. Specifically, we introduce the Activation Addition (ActAdd) technique, which contrasts the intermediate activations on prompt pairs (such as "Love" versus "Hate") to compute a steering vector (Subramani et al. 2022). By tactically adding in e.g. the "Love" - "Hate" steering vector during the forward pass, we achieve SOTA on negative-to-positive sentiment shift and detoxification using models including LLaMA-3 and OPT. ActAdd yields inference-time control over high-level output properties (like topic and sentiment) while preserving performance on off-target tasks. ActAdd is lightweight: it does not require any machine optimization and works with a single pair of data points, which enables rapid iteration over steering. ActAdd demonstrates the power of activation engineering.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2308.10248>
- PDF: [[raw/papers/pdf/2023-steering-language-models-with-activation-engineering.pdf]]
- Raw markdown: [[raw/papers/md/2023-steering-language-models-with-activation-engineering]]
