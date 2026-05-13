---
arxiv: '2408.15664'
authors:
- Lean Wang
- Huazuo Gao
- Chenggang Zhao
- Xu Sun
- Damai Dai
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2408.15664
  raw: '[[raw/papers/md/2024-auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts]]'
  source: https://arxiv.org/abs/2408.15664
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts.md
raw_pdf: raw/papers/pdf/2024-auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts.pdf
read: false
slug: auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts
tags:
- type/paper
- status/stub
title: Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
type: note
updated: '2026-05-11'
year: 2024
---

# Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

> *Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

For Mixture-of-Experts (MoE) models, an unbalanced expert load will lead to routing collapse or increased computational overhead. Existing methods commonly employ an auxiliary loss to encourage load balance, but a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance. In order to control load balance while not producing undesired gradients during training, we propose Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy. To be specific, before the top-K routing decision, Loss-Free Balancing will first apply an expert-wise bias to the routing scores of each expert. By dynamically updating the bias of each expert according to its recent load, Loss-Free Balancing can consistently maintain a balanced distribution of expert load. In addition, since Loss-Free Balancing does not produce any interference gradients, it also elevates the upper bound of model performance gained from MoE training. We validate the performance of Loss-Free Balancing on MoE models with up to 3B parameters trained on up to 200B tokens. Experimental results show that Loss-Free Balancing achieves both better performance and better load balance compared with traditional auxiliary-loss-controlled load balancing strategies.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2408.15664>
- PDF: [[raw/papers/pdf/2024-auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts.pdf]]
- Raw markdown: [[raw/papers/md/2024-auxiliary-loss-free-load-balancing-strategy-for-mixture-of-experts]]
