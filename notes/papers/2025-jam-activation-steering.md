---
arxiv: '2502.20684'
authors:
- Yingbing Huang
- Deming Chen
- Abhishek K. Umrawal
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.20684
  raw: null
  source: https://arxiv.org/abs/2502.20684
owner: blaz
raw_pdf: raw/papers/pdf/2025-jam-activation-steering.pdf
read: false
slug: jam-activation-steering
tags:
- type/paper
- source/primary
- status/stub
- domain/inference
- domain/llm
title: 'JAM: Controllable and Responsible Text Generation via Causal Reasoning and
  Latent Vector Manipulation'
type: note
updated: '2026-05-10'
year: 2025
---

# JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation

Latent-space intervention that integrates causal reasoning to edit activation vectors during the forward pass, steering attributes without retraining.

- **Authors**: Yingbing Huang, Deming Chen, Abhishek K. Umrawal
- **Venue**: arXiv 2025
- **arXiv**: [2502.20684](https://arxiv.org/abs/2502.20684)
- **Raw**: [[raw/papers/pdf/2025-jam-activation-steering]]

## Core contribution

JAM integrates causal reasoning within the latent space of LLMs to control text generation at inference time. Four-step forward-pass procedure: latent vector extraction, attribute detection, manipulation vector generation, and manipulated generation update. Steers attributes such as safety/toxicity without retraining.
