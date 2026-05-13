---
authors:
- Xian Li
- Ping Yu
- Chunting Zhou
- Timo Schick
- Omer Levy
- Luke Zettlemoyer
- Jason Weston
- Mike Lewis
created: '2026-05-12'
kind: paper
links:
  code: null
  paper: https://openreview.net/pdf?id=1oijHJBRsT
  raw: '[[raw/papers/md/2024-self-alignment-with-instruction-backtranslation]]'
  source: https://openreview.net/pdf?id=1oijHJBRsT
owner: blaz
parser: read
raw_md: raw/papers/md/2024-self-alignment-with-instruction-backtranslation.md
raw_pdf: raw/papers/pdf/2024-self-alignment-with-instruction-backtranslation.pdf
read: false
slug: self-alignment-with-instruction-backtranslation
tags:
- type/paper
- status/stub
title: Self-Alignment with Instruction Backtranslation
type: note
updated: '2026-05-12'
venue: ICLR 2024
year: 2024
---

# Self-Alignment with Instruction Backtranslation

> *Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, et al.* — Meta, ICLR 2024

## TL;DR

(stub — fill in after reading)

## Abstract

We present a scalable method to build a high quality instruction following language model by automatically labelling human-written text with corresponding instructions. Our approach, named instruction backtranslation, starts with a language model finetuned on a small amount of seed data, and a given web corpus. The seed model is used to construct training examples by generating instruction prompts for web documents (self-augmentation), and then selecting high quality examples from among these candidates (self-curation). This data is then used to finetune a stronger model. Finetuning LLaMa on two iterations of our approach yields a model that outperforms all other LLaMA-based models on the Alpaca leaderboard not relying on distillation data, demonstrating highly effective self-alignment.

## Notes

(stub)

## Source

- OpenReview: <https://openreview.net/pdf?id=1oijHJBRsT>
- PDF: [[raw/papers/pdf/2024-self-alignment-with-instruction-backtranslation.pdf]]
