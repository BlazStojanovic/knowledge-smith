---
arxiv: '2306.03341'
authors:
- Kenneth Li
- Oam Patel
- Fernanda Viégas
- Hanspeter Pfister
- Martin Wattenberg
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2306.03341
  raw: '[[raw/papers/md/2023-inference-time-intervention-eliciting-truthful-answers-from-a-language-model]]'
  source: https://arxiv.org/abs/2306.03341
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-inference-time-intervention-eliciting-truthful-answers-from-a-language-model.md
raw_pdf: raw/papers/pdf/2023-inference-time-intervention-eliciting-truthful-answers-from-a-language-model.pdf
read: false
slug: inference-time-intervention-eliciting-truthful-answers-from-a-language-model
tags:
- type/paper
- status/stub
title: 'Inference-Time Intervention: Eliciting Truthful Answers from a Language Model'
type: note
updated: '2026-05-11'
year: 2023
---

# Inference-Time Intervention: Eliciting Truthful Answers from a Language Model

> *Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

We introduce Inference-Time Intervention (ITI), a technique designed to enhance the "truthfulness" of large language models (LLMs). ITI operates by shifting model activations during inference, following a set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from 32.5% to 65.1%. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2306.03341>
- PDF: [[raw/papers/pdf/2023-inference-time-intervention-eliciting-truthful-answers-from-a-language-model.pdf]]
- Raw markdown: [[raw/papers/md/2023-inference-time-intervention-eliciting-truthful-answers-from-a-language-model]]
