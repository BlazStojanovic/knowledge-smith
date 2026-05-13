---
arxiv: '2506.10943'
authors:
- Adam Zweiger
- Jyothish Pari
- Han Guo
- Ekin Akyürek
- Yoon Kim
- Pulkit Agrawal
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.10943
  raw: '[[raw/papers/md/2025-self-adapting-language-models]]'
  source: https://arxiv.org/abs/2506.10943
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-self-adapting-language-models.md
raw_pdf: raw/papers/pdf/2025-self-adapting-language-models.pdf
read: false
slug: self-adapting-language-models
tags:
- type/paper
- status/stub
title: Self-Adapting Language Models
type: note
updated: '2026-05-11'
year: 2025
---

# Self-Adapting Language Models

> *Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models (LLMs) are powerful but static; they lack mechanisms to adapt their weights in response to new tasks, knowledge, or examples. We introduce Self-Adapting LLMs (SEAL), a framework that enables LLMs to self-adapt by generating their own finetuning data and update directives. Given a new input, the model produces a self-edit-a generation that may restructure the information in different ways, specify optimization hyperparameters, or invoke tools for data augmentation and gradient-based updates. Through supervised finetuning (SFT), these self-edits result in persistent weight updates, enabling lasting adaptation. To train the model to produce effective self-edits, we use a reinforcement learning loop with the downstream performance of the updated model as the reward signal. Unlike prior approaches that rely on separate adaptation modules or auxiliary networks, SEAL directly uses the model's own generation to control its adaptation process. Experiments on knowledge incorporation and few-shot generalization show that SEAL is a promising step toward language models capable of self-directed adaptation. Our website and code is available at https://jyopari.github.io/posts/seal.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.10943>
- PDF: [[raw/papers/pdf/2025-self-adapting-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2025-self-adapting-language-models]]
