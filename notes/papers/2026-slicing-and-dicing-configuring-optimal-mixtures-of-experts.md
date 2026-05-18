---
arxiv: '2605.11689'
authors:
- Margaret Li
- Sneha Kudugunta
- Danielle Rothermel
- Luke Zettlemoyer
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.11689
  raw: '[[raw/papers/md/2026-slicing-and-dicing-configuring-optimal-mixtures-of-experts]]'
  source: https://arxiv.org/abs/2605.11689
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-slicing-and-dicing-configuring-optimal-mixtures-of-experts.md
raw_pdf: raw/papers/pdf/2026-slicing-and-dicing-configuring-optimal-mixtures-of-experts.pdf
read: false
slug: slicing-and-dicing-configuring-optimal-mixtures-of-experts
tags:
- type/paper
- status/stub
- mixture-of-experts
- pretraining
- scaling-laws
- architecture
title: 'Slicing and Dicing: Configuring Optimal Mixtures of Experts'
type: note
updated: '2026-05-18'
year: 2026
---

# Slicing and Dicing: Configuring Optimal Mixtures of Experts

> *Margaret Li, Sneha Kudugunta, Danielle Rothermel, Luke Zettlemoyer* — arXiv 2026

## TL;DR

First systematic large-scale study of MoE design choices: **2,000+ pretraining runs**, models up to 6.6B total params, exhaustively varying total expert count, expert dimension, heterogeneous expert sizing within a layer, shared-expert size, and load-balancing mechanism. Three findings: (1) at every active-parameter scale, performance keeps improving with total MoE parameters even at extreme active:total ratios (e.g. 128×); (2) optimal expert size is nearly **invariant to total parameter count** — it depends only on active parameter count; (3) shared experts, heterogeneous experts, and load-balancing settings have small effects relative to expert count and granularity, though dropless routing gives a consistent gain. Recipe: focus on expert count and granularity; the other knobs barely move final quality. (Summary from abstract; note unread.)

## Abstract

Mixture-of-Experts (MoE) architectures have become standard in large language models, yet many of their core design choices - expert count, granularity, shared experts, load balancing, token dropping - have only been studied one or two at a time over narrow configuration ranges. It remains an open question whether these choices can be optimized independently, without considering interactions. We present the first systematic study of over 2,000 pretraining runs spanning models up to 6.6B total parameters, in which we exhaustively vary total experts, expert dimension, heterogeneous expert sizing within a single layer, shared expert size and load-balancing mechanisms. We find that at every active-parameter scale that we study, performance consistently improves with total MoE parameters even at extreme active expert parameter ratios like 128.Further, the optimal expert size is nearly invariant to total parameter count and depends only on active parameter count. Third, we see that other choices like shared experts, heterogeneous experts and load-balancing settings have small effects relative to expert count and granularity, although dropless routing yields a consistent gain. Overall, our results suggest a simpler recipe: focus on expert count and granularity, other choices have minimal effect on final quality.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.11689>
- PDF: [[raw/papers/pdf/2026-slicing-and-dicing-configuring-optimal-mixtures-of-experts.pdf]]
- Raw markdown: [[raw/papers/md/2026-slicing-and-dicing-configuring-optimal-mixtures-of-experts]]
