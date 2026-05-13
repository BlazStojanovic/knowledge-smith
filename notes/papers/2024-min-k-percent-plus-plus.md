---
arxiv: '2404.02936'
authors:
- Jingyang Zhang
- Jingwei Sun
- Eric Yeats
- Yang Ouyang
- Martin Kuo
- Jianyi Zhang
- Hao Frank Yang
- Hai Li
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.02936
  raw: null
  source: https://arxiv.org/abs/2404.02936
owner: blaz
raw_pdf: raw/papers/pdf/2024-min-k-percent-plus-plus.pdf
read: false
slug: min-k-percent-plus-plus
tags:
- type/paper
- source/primary
- status/stub
- domain/evals
title: 'Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language
  Models'
type: note
updated: '2026-05-10'
year: 2024
---

# Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models

Theoretically-grounded improvement to Min-K% Prob: frames detection as identifying modes (local maxima) of the model's conditional categorical distribution along input dimensions.

- **Authors**: Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, Hai Li
- **Venue**: ICLR 2025 (Spotlight)
- **arXiv**: [2404.02936](https://arxiv.org/abs/2404.02936)
- **Raw**: [[raw/papers/pdf/2024-min-k-percent-plus-plus]]

## Core contribution

Improves on [[notes/papers/2024-min-k-percent]] by providing a theoretically motivated detection criterion. Training samples tend to be local maxima of the modeled distribution; Min-K%++ evaluates whether input sequences form modes under the model's conditional categorical distribution.

## Key results

- 6.2% to 10.5% AUROC improvement over Min-K% on WikiMIA benchmark
- Competitive performance on MIMIR benchmark

## Connections

- Map: [[maps/model-evaluation/contamination-methods]] — gray-box, likelihood-based signal
- Concept: [[concepts/benchmark-contamination]]
- Predecessor: [[notes/papers/2024-min-k-percent]]
