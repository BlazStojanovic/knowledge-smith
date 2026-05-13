---
arxiv: '2603.21191'
authors:
- Rustem Islamov
- Roman Machacek
- Aurelien Lucchi
- Antonio Silveti-Falls
- Eduard Gorbunov
- Volkan Cevher
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.21191
  raw: '[[raw/papers/md/2026-on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods]]'
  source: https://arxiv.org/abs/2603.21191
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods.md
raw_pdf: raw/papers/pdf/2026-on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods.pdf
read: false
slug: on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods
tags:
- type/paper
- status/stub
title: On the Role of Batch Size in Stochastic Conditional Gradient Methods
type: note
updated: '2026-05-11'
year: 2026
---

# On the Role of Batch Size in Stochastic Conditional Gradient Methods

> *Rustem Islamov, Roman Machacek, Aurelien Lucchi, Antonio Silveti-Falls, Eduard Gorbunov, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We study the role of batch size in stochastic conditional gradient methods under a $μ$-Kurdyka-Łojasiewicz ($μ$-KL) condition. Focusing on momentum-based stochastic conditional gradient algorithms (e.g., Scion), we derive a new analysis that explicitly captures the interaction between stepsize, batch size, and stochastic noise. Our study reveals a regime-dependent behavior: increasing the batch size initially improves optimization accuracy but, beyond a critical threshold, the benefits saturate and can eventually degrade performance under a fixed token budget. Notably, the theory predicts the magnitude of the optimal stepsize and aligns well with empirical practices observed in large-scale training. Leveraging these insights, we derive principled guidelines for selecting the batch size and stepsize, and propose an adaptive strategy that increases batch size and sequence length during training while preserving convergence guarantees. Experiments on NanoGPT are consistent with the theoretical predictions and illustrate the emergence of the predicted scaling regimes. Overall, our results provide a theoretical framework for understanding batch size scaling in stochastic conditional gradient methods and offer guidance for designing efficient training schedules in large-scale optimization.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.21191>
- PDF: [[raw/papers/pdf/2026-on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods.pdf]]
- Raw markdown: [[raw/papers/md/2026-on-the-role-of-batch-size-in-stochastic-conditional-gradient-methods]]
