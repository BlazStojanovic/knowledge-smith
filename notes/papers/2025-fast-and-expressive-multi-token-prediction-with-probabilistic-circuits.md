---
arxiv: '2511.11346'
authors:
- Andreas Grivas
- Lorenzo Loconte
- Emile van Krieken
- Piotr Nawrot
- Yu Zhao
- Euan Wielewski
- Pasquale Minervini
- Edoardo Ponti
- Antonio Vergari
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2511.11346
  raw: '[[raw/papers/md/2025-fast-and-expressive-multi-token-prediction-with-probabilistic-circuits]]'
  source: https://arxiv.org/abs/2511.11346
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-fast-and-expressive-multi-token-prediction-with-probabilistic-circuits.md
raw_pdf: raw/papers/pdf/2025-fast-and-expressive-multi-token-prediction-with-probabilistic-circuits.pdf
read: false
slug: fast-and-expressive-multi-token-prediction-with-probabilistic-circuits
tags:
- type/paper
- status/stub
title: Fast and Expressive Multi-Token Prediction with Probabilistic Circuits
type: note
updated: '2026-05-11'
year: 2025
---

# Fast and Expressive Multi-Token Prediction with Probabilistic Circuits

> *Andreas Grivas, Lorenzo Loconte, Emile van Krieken, Piotr Nawrot, Yu Zhao, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Multi-token prediction (MTP) is a prominent strategy to significantly speed up generation in large language models (LLMs), including byte-level LLMs, which are tokeniser-free but prohibitively slow. However, existing MTP methods often sacrifice expressiveness by assuming independence between future tokens. In this work, we investigate the trade-off between expressiveness and latency in MTP within the framework of probabilistic circuits (PCs). Our framework, named MTPC, allows one to explore different ways to encode the joint distributions over future tokens by selecting different circuit architectures, generalising classical models such as (hierarchical) mixture models, hidden Markov models and tensor networks. We show the efficacy of MTPC by retrofitting existing byte-level LLMs, such as EvaByte. Our experiments show that, when combined with speculative decoding, MTPC significantly speeds up generation compared to MTP with independence assumptions, while guaranteeing to retain the performance of the original verifier LLM. We also rigorously study the optimal trade-off between expressiveness and latency when exploring the possible parameterisations of MTPC, such as PC architectures and partial layer sharing between the verifier and draft LLMs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2511.11346>
- PDF: [[raw/papers/pdf/2025-fast-and-expressive-multi-token-prediction-with-probabilistic-circuits.pdf]]
- Raw markdown: [[raw/papers/md/2025-fast-and-expressive-multi-token-prediction-with-probabilistic-circuits]]
