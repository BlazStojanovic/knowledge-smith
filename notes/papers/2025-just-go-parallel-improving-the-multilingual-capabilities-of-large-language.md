---
arxiv: '2506.13044'
authors:
- Muhammad Reza Qorib
- Junyi Li
- Hwee Tou Ng
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.13044
  raw: '[[raw/papers/md/2025-just-go-parallel-improving-the-multilingual-capabilities-of-large-language]]'
  source: https://arxiv.org/abs/2506.13044
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-just-go-parallel-improving-the-multilingual-capabilities-of-large-language.md
raw_pdf: raw/papers/pdf/2025-just-go-parallel-improving-the-multilingual-capabilities-of-large-language.pdf
read: false
slug: just-go-parallel-improving-the-multilingual-capabilities-of-large-language
tags:
- type/paper
- status/stub
- multilingual
- pretraining
- llm
title: 'Just Go Parallel: Improving the Multilingual Capabilities of Large Language
  Models'
type: note
updated: '2026-05-15'
year: 2025
---

# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models

> *Muhammad Reza Qorib, Junyi Li, Hwee Tou Ng* — arXiv 2025

## TL;DR

Parallel data is still valuable for decoder LLMs. Translation pairs (parallel data) improve translation and multilingual commonsense reasoning more than unrelated multilingual monolingual text. The best recipe is to add parallel data late, as a second-stage training phase, because adding it early can get overwritten. **Note**: directional training matters; models do not automatically learn reverse translation just because they saw one direction. **Caveat**: No evals on maths/coding, only knowledge/reasoning. (tl;dr by Blaz.)

## Abstract

Large language models (LLMs) have demonstrated impressive translation capabilities even without being explicitly trained on parallel data. This remarkable property has led some to believe that parallel data is no longer necessary for building multilingual language models. While some attribute this to the emergent abilities of LLMs due to scale, recent work suggests that it is actually caused by incidental bilingual signals present in the training data. Various methods have been proposed to maximize the utility of parallel data to enhance the multilingual capabilities of multilingual encoder-based and encoder-decoder language models. However, some decoder-based LLMs opt to ignore parallel data instead. In this work, we conduct a systematic study on the impact of adding parallel data on LLMs' multilingual capabilities, focusing specifically on translation and multilingual common-sense reasoning. Through controlled experiments, we demonstrate that parallel data can significantly improve LLMs' multilingual capabilities.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.13044>
- PDF: [[raw/papers/pdf/2025-just-go-parallel-improving-the-multilingual-capabilities-of-large-language.pdf]]
- Raw markdown: [[raw/papers/md/2025-just-go-parallel-improving-the-multilingual-capabilities-of-large-language]]
