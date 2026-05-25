---
arxiv: '2310.01334'
authors:
- Pingzhi Li
- Zhenyu Zhang
- Prateek Yadav
- Yi-Lin Sung
- Yu Cheng
- Mohit Bansal
- Tianlong Chen
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2310.01334
  raw: '[[raw/papers/md/2023-merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy]]'
  source: https://arxiv.org/abs/2310.01334
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy.md
raw_pdf: raw/papers/pdf/2023-merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy.pdf
read: false
slug: merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy
tags:
- type/paper
- status/stub
title: 'Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing
  Policy'
type: note
updated: '2026-05-25'
year: 2023
---

# Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy

> *Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, et al.* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

Sparsely activated Mixture-of-Experts (SMoE) has shown promise to scale up the learning capacity of neural networks, however, they have issues like (a) High Memory Usage, due to duplication of the network layers into multiple copies as experts; and (b) Redundancy in Experts, as common learning-based routing policies suffer from representational collapse. Therefore, vanilla SMoE models are memory inefficient and non-scalable, especially for resource-constrained downstream scenarios. In this paper, we ask: Can we craft a compact SMoE model by consolidating expert information? What is the best recipe to merge multiple experts into fewer but more knowledgeable experts? Our pilot investigation reveals that conventional model merging methods fail to be effective in such expert merging for SMoE. The potential reasons are: (1) redundant information overshadows critical experts; (2) appropriate neuron permutation for each expert is missing to bring all of them in alignment. To address this, we propose M-SMoE, which leverages routing statistics to guide expert merging. Specifically, it starts with neuron permutation alignment for experts; then, dominant experts and their "group members" are formed; lastly, every expert group is merged into a single expert by utilizing each expert's activation frequency as their weight for merging, thus diminishing the impact of insignificant experts. Moreover, we observed that our proposed merging promotes a low dimensionality in the merged expert's weight space, naturally paving the way for additional compression. Hence, our final method, MC-SMoE (i.e., Merge, then Compress SMoE), further decomposes the merged experts into low-rank and structural sparse alternatives. Extensive experiments across 8 benchmarks validate the effectiveness of MC-SMoE. For instance, our MC-SMoE achieves up to 80% memory and a 20% FLOPs reduction, with virtually no loss in performance.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2310.01334>
- PDF: [[raw/papers/pdf/2023-merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy.pdf]]
- Raw markdown: [[raw/papers/md/2023-merge-then-compress-demystify-efficient-smoe-with-hints-from-its-routing-policy]]
