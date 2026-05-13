---
arxiv: '2603.12273'
authors:
- Thomas Kleine Buening
- Jonas Hübotter
- Barna Pásztor
- Idan Shenfeld
- Giorgia Ramponi
- Andreas Krause
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.12273
  raw: '[[raw/papers/md/2026-aligning-language-models-from-user-interactions]]'
  source: https://arxiv.org/abs/2603.12273
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-aligning-language-models-from-user-interactions.md
raw_pdf: raw/papers/pdf/2026-aligning-language-models-from-user-interactions.pdf
read: false
slug: aligning-language-models-from-user-interactions
tags:
- type/paper
- status/stub
title: Aligning Language Models from User Interactions
type: note
updated: '2026-05-11'
year: 2026
---

# Aligning Language Models from User Interactions

> *Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user's preferences. Importantly, language models are already able to make use of this information in context. After observing a user's follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user's follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model's behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.12273>
- PDF: [[raw/papers/pdf/2026-aligning-language-models-from-user-interactions.pdf]]
- Raw markdown: [[raw/papers/md/2026-aligning-language-models-from-user-interactions]]
