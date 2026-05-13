---
authors:
- Amanda Bertsch
- Luca Soldaini
- Matthew R. Gormley
- Graham Neubig
- Hannaneh Hajishirzi
- Kyle Lo
- Dirk Groeneveld
created: '2026-05-12'
kind: paper
links:
  code: null
  paper: https://allenai.org/papers/olmpool
  raw: '[[raw/papers/md/2026-cracks-in-the-foundation-architectural-choices-long-context]]'
  source: https://allenai.org/papers/olmpool
owner: blaz
parser: read
raw_md: raw/papers/md/2026-cracks-in-the-foundation-architectural-choices-long-context.md
raw_pdf: raw/papers/pdf/2026-cracks-in-the-foundation-architectural-choices-long-context.pdf
read: false
slug: cracks-in-the-foundation-architectural-choices-long-context
tags:
- type/paper
- status/stub
title: 'Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long
  Context Extension'
type: note
updated: '2026-05-12'
venue: preprint
year: 2026
---

# Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension

> *Amanda Bertsch, Luca Soldaini, Matthew R. Gormley, Graham Neubig, Hannaneh Hajishirzi, Kyle Lo, Dirk Groeneveld* — Allen Institute for AI / CMU / U. Washington, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

One might imagine that architectural variations within the dense transformer paradigm have a limited effect on accuracy. However, we demonstrate that this is not the case in the long context setting. Specifically, we show that a set of four minor architectural decisions—all made by at least one of the Olmo, Llama, and Qwen dense model families—have a compoundingly negative effect on long context extensibility. Any one of these choices alone has a minor impact on long context performance, but combining three or more can drop the performance downstream by up to 47%. Furthermore, these differences are not detectable from short-context loss or validation datasets. We demonstrate this with controlled ablations that hold data, tokenizer, and extension recipe fixed while varying normalization, GQA, pretraining context length, and sliding window attention. We show that much of the variation in long context ability across model families is driven by these architectural features and detectable from applying context extension early in pretraining. After over 170,000 GPU hours of training, we release the resulting set of models as OlmPool, a set of 26 comparable 7B models with checkpoints before and after long-context extension. This pool includes several architectures that outperform the Llama 3 architecture on long context extensibility. In an analysis of our ablation models, we identify patterns in attention sink behavior and attention distributions across context that are attributable to specific architectural differences.

## Notes

(stub)

## Source

- Tech report PDF: <https://allenai.org/papers/olmpool>
- PDF: [[raw/papers/pdf/2026-cracks-in-the-foundation-architectural-choices-long-context.pdf]]
