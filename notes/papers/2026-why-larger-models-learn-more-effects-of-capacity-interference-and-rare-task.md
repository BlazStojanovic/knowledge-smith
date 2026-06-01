---
arxiv: '2605.29548'
authors:
- Jing Huang
- Daniel Wurgaft
- Rachit Bansal
- Laura Ruis
- Naomi Saphra
- David Alvarez-Melis
- Andrew Kyle Lampinen
- Christopher Potts
- Ekdeep Singh Lubana
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.29548
  raw: '[[raw/papers/md/2026-why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task]]'
  source: https://arxiv.org/abs/2605.29548
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task.md
raw_pdf: raw/papers/pdf/2026-why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task.pdf
read: false
slug: why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task
tags:
- type/paper
- status/stub
title: 'Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task
  Retention'
type: note
updated: '2026-06-01'
year: 2026
---

# Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention

> *Jing Huang, Daniel Wurgaft, Rachit Bansal, Laura Ruis, Naomi Saphra, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Larger models learn tasks smaller models do not. What drives this phenomenon? We develop a simple phenomenological argument that power-law scaling already suggests that a larger model will be able to learn a part of the data distribution that a smaller model fails to learn, even with infinite training data. To validate this claim and identify its causes, we study the effects of model scaling on a synthetic setup consisting of a mixture of tasks that show monotonic scaling curves. The results point to a data-induced competition over resources (neurons). Specifically, smaller models allocate their neurons to high frequency or low complexity tasks, and so they learn solutions that perform poorly on rare and complex tasks. Moreover, this happens even when solutions capable of expressing the desired task exist. We then assess how a larger model circumvents this data-centric bottleneck, finding that it traces to a reduced interference mechanism: larger models can allocate enough resources to common tasks that the gradient updates for those tasks become weak, which means that they do not overwrite rare-task features as they slowly accumulate. Finally, to further validate these claims, we pretrain OLMo models (4M to 4B parameters) on novel tasks of varying frequency and complexity. The results mirror those from our synthetic data experiments: only the larger OLMo models learn the infrequent and complex tasks, and these larger models embed more task features in their representations and show less gradient interference between tasks. Overall, we offer a data-centric account of why larger models learn tasks that smaller models fail to. This helps explain why larger models are better in practice, and it can inform practical questions concerning model sizing and training data mixtures.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.29548>
- PDF: [[raw/papers/pdf/2026-why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task.pdf]]
- Raw markdown: [[raw/papers/md/2026-why-larger-models-learn-more-effects-of-capacity-interference-and-rare-task]]
