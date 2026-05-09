---
arxiv: '2309.14322'
authors:
- Mitchell Wortsman
- Peter J. Liu
- Lechao Xiao
- Katie Everett
- Alex Alemi
- Ben Adlam
- John D. Co-Reyes
- Izzeddin Gur
- Abhishek Kumar
- Roman Novak
- Jeffrey Pennington
- Jascha Sohl-dickstein
- Kelvin Xu
- Jaehoon Lee
- Justin Gilmer
- Simon Kornblith
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2023-small-scale-proxies-for-large-scale-transformer-training.md
raw_pdf: raw/papers/pdf/2023-small-scale-proxies-for-large-scale-transformer-training.pdf
read: false
slug: small-scale-proxies-for-large-scale-transformer-training
tags:
- transformer
- pretraining
- scaling-laws
- optimization
title: Small-scale proxies for large-scale Transformer training instabilities
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2309.14322
venue: null
year: 2023
---

# Small-scale proxies for large-scale Transformer training instabilities

> *Mitchell Wortsman, Peter J. Liu, Lechao Xiao…* — arXiv 2309.14322, 2023

## TL;DR

(stub — fill in after reading)

## Abstract

Teams that have trained large Transformer-based models have reported training instabilities at large scale that did not appear when training with the same hyperparameters at smaller scales. Although the causes of such instabilities are of scientific interest, the amount of resources required to reproduce them has made investigation difficult. In this work, we seek ways to reproduce and study training stability and instability at smaller scales. First, we focus on two sources of training instability described in previous work: the growth of logits in attention layers (Dehghani et al., 2023) and divergence of the output logits from the log probabilities (Chowdhery et al., 2022). By measuring the relationship between learning rate and loss across scales, we show that these instabilities also appear in small models when training at high learning rates, and that mitigations previously employed at large scales are equally effective in this regime. This prompts us to investigate the extent to which other known optimizer and model interventions influence the sensitivity of the final loss to changes in the learning rate. To this end, we study methods such as warm-up, weight decay, and the μParam (Yang et al., 2022), and combine techniques to train small models that achieve similar losses across orders of magnitude of learning rate variation. Finally, to conclude our exploration we study two cases where instabilities can be predicted before they emerge by examining the scaling behavior of model activation and gradient norms.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2023-small-scale-proxies-for-large-scale-transformer-training]]
- PDF: [[raw/papers/pdf/2023-small-scale-proxies-for-large-scale-transformer-training.pdf]]
- arXiv: <https://arxiv.org/abs/2309.14322>
