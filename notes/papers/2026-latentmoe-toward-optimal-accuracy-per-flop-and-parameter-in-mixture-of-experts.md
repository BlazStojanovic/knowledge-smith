---
arxiv: '2601.18089'
authors:
- Venmugil Elango
- Nidhi Bhatia
- Roger Waleffe
- Rasoul Shafipour
- Tomer Asida
- Abhinav Khattar
- Nave Assaf
- Maximilian Golub
- Joey Guman
- Tiyasa Mitra
- Ritchie Zhao
- Ritika Borkar
- Ran Zilberstein
- Mostofa Patwary
- Mohammad Shoeybi
- Bita Rouhani
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.18089
  raw: '[[raw/papers/md/2026-latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts]]'
  source: https://arxiv.org/abs/2601.18089
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts.md
raw_pdf: raw/papers/pdf/2026-latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts.pdf
read: false
slug: latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts
tags:
- type/paper
- status/stub
title: 'LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts'
type: note
updated: '2026-05-11'
year: 2026
---

# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

> *Venmugil Elango, Nidhi Bhatia, Roger Waleffe, Rasoul Shafipour, Tomer Asida, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Mixture of Experts (MoEs) have become a central component of many state-of-the-art open-source and proprietary large language models. Despite their widespread adoption, it remains unclear how close existing MoE architectures are to optimal with respect to inference cost, as measured by accuracy per floating-point operation and per parameter. In this work, we revisit MoE design from a hardware-software co-design perspective, grounded in empirical and theoretical considerations. We characterize key performance bottlenecks across diverse deployment regimes, spanning offline high-throughput execution and online, latency-critical inference. Guided by these insights, we introduce LatentMoE, a new model architecture resulting from systematic design exploration and optimized for maximal accuracy per unit of compute. Empirical design space exploration at scales of up to 95B parameters and over a 1T-token training horizon, together with supporting theoretical analysis, shows that LatentMoE consistently outperforms standard MoE architectures in terms of accuracy per FLOP and per parameter. Given its strong performance, the LatentMoE architecture has been adopted by the flagship Nemotron-3 Super and Ultra models and scaled to substantially larger regimes, including longer token horizons and larger model sizes, as reported in Nvidia et al. (arXiv:2512.20856).

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.18089>
- PDF: [[raw/papers/pdf/2026-latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts.pdf]]
- Raw markdown: [[raw/papers/md/2026-latentmoe-toward-optimal-accuracy-per-flop-and-parameter-in-mixture-of-experts]]
