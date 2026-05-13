---
arxiv: '2512.09015'
authors:
- DatologyAI
- ':'
- Luke Merrick
- Alex Fang
- Aldo Carranza
- Alvin Deng
- Amro Abbas
- Brett Larsen
- Cody Blakeney
- Darren Teh
- David Schwab
- Fan Pan
- Haakon Mongstad
- Haoli Yin
- Jack Urbanek
- Jason Lee
- Jason Telanoff
- Josh Wills
- Kaleigh Mentzer
- Paul Burstein
- Parth Doshi
- Paul Burnstein
- Pratyush Maini
- Ricardo Monti
- Rishabh Adiga
- Scott Loftin
- Siddharth Joshi
- Spandan Das
- Tony Jiang
- Vineeth Dorna
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.09015
  raw: '[[raw/papers/md/2025-luxical-high-speed-lexical-dense-text-embeddings]]'
  source: https://arxiv.org/abs/2512.09015
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-luxical-high-speed-lexical-dense-text-embeddings.md
raw_pdf: raw/papers/pdf/2025-luxical-high-speed-lexical-dense-text-embeddings.pdf
read: false
slug: luxical-high-speed-lexical-dense-text-embeddings
tags:
- type/paper
- status/stub
title: 'Luxical: High-Speed Lexical-Dense Text Embeddings'
type: note
updated: '2026-05-11'
year: 2025
---

# Luxical: High-Speed Lexical-Dense Text Embeddings

> *DatologyAI, :, Luke Merrick, Alex Fang, Aldo Carranza, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a concrete Luxical model in two disparate applications: a targeted webcrawl document retrieval test and an end-to-end language model data curation task grounded in text classification. In these tasks we demonstrate speedups ranging from 3x to 100x over varying-sized neural baselines, and comparable to FastText model inference during the data curation task. On these evaluations, the tested Luxical model illustrates favorable compute/quality trade-offs for large-scale text organization, matching the quality of neural baselines. Luxical is available as open-source software at https://github.com/datologyai/luxical.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.09015>
- PDF: [[raw/papers/pdf/2025-luxical-high-speed-lexical-dense-text-embeddings.pdf]]
- Raw markdown: [[raw/papers/md/2025-luxical-high-speed-lexical-dense-text-embeddings]]
