---
arxiv: '1701.06538'
authors:
- Noam Shazeer
- Azalia Mirhoseini
- Krzysztof Maziarz
- Andy Davis
- Quoc Le
- Geoffrey Hinton
- Jeff Dean
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1701.06538
  raw: '[[raw/papers/md/2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer]]'
  source: https://arxiv.org/abs/1701.06538
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer.md
raw_pdf: raw/papers/pdf/2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer.pdf
read: false
slug: outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer
tags:
- type/paper
- status/stub
title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
  Layer'
type: note
updated: '2026-05-11'
year: 2017
---

# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

> *Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, et al.* — arXiv 2017

## TL;DR

(stub — fill in after reading)

## Abstract

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/1701.06538>
- PDF: [[raw/papers/pdf/2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer.pdf]]
- Raw markdown: [[raw/papers/md/2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer]]
