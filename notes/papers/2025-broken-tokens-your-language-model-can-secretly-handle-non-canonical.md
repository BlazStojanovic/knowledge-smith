---
arxiv: '2506.19004'
authors:
- Brian Siyuan Zheng
- Alisa Liu
- Orevaoghene Ahia
- Jonathan Hayase
- Yejin Choi
- Noah A. Smith
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.19004
  raw: '[[raw/papers/md/2025-broken-tokens-your-language-model-can-secretly-handle-non-canonical]]'
  source: https://arxiv.org/abs/2506.19004
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-broken-tokens-your-language-model-can-secretly-handle-non-canonical.md
raw_pdf: raw/papers/pdf/2025-broken-tokens-your-language-model-can-secretly-handle-non-canonical.pdf
read: false
slug: broken-tokens-your-language-model-can-secretly-handle-non-canonical
tags:
- type/paper
- status/stub
title: Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations
type: note
updated: '2026-05-11'
year: 2025
---

# Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations

> *Brian Siyuan Zheng, Alisa Liu, Orevaoghene Ahia, Jonathan Hayase, Yejin Choi, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Modern tokenizers employ deterministic algorithms to map text into a single "canonical" token sequence, yet the same string can be encoded as many non-canonical tokenizations using the tokenizer vocabulary. In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training. Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization. We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form. Motivated by these results, we then identify settings where non-canonical tokenization schemes can *improve* performance, finding that character-level segmentation improves string manipulation and code understanding tasks by up to +14%, and right-aligned digit grouping enhances large-number arithmetic by +33%. Finally, we investigate the source of this robustness, finding that it arises in the instruction-tuning phase. We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses. Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.19004>
- PDF: [[raw/papers/pdf/2025-broken-tokens-your-language-model-can-secretly-handle-non-canonical.pdf]]
- Raw markdown: [[raw/papers/md/2025-broken-tokens-your-language-model-can-secretly-handle-non-canonical]]
