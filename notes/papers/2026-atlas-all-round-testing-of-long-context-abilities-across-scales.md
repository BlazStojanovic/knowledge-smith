---
arxiv: '2605.28079'
authors:
- Deli Huang
- Cunguang Wang
- Hongyin Tang
- Zhe Tang
- Linsen Guo
- Dongyu Ru
- Ruoshi Yuan
- Ziyue Zhu
- Xiaoyu Li
- Ziwen Wang
- Chen Zhang
- Anchun Gui
- Wen Zan
- Jiaqi Zhang
- Xuezhi Cao
- Jingang Wang
- Xunliang Cai
- Yixin Cao
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.28079
  raw: '[[raw/papers/md/2026-atlas-all-round-testing-of-long-context-abilities-across-scales]]'
  source: https://arxiv.org/abs/2605.28079
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-atlas-all-round-testing-of-long-context-abilities-across-scales.md
raw_pdf: raw/papers/pdf/2026-atlas-all-round-testing-of-long-context-abilities-across-scales.pdf
read: false
slug: atlas-all-round-testing-of-long-context-abilities-across-scales
tags:
- type/paper
- status/stub
title: 'ATLAS: All-round Testing of Long-context Abilities across Scales'
type: note
updated: '2026-06-01'
year: 2026
---

# ATLAS: All-round Testing of Long-context Abilities across Scales

> *Deli Huang, Cunguang Wang, Hongyin Tang, Zhe Tang, Linsen Guo, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Long-context language models now advertise context windows up to millions of tokens, yet evaluations typically report a single length or a narrow task family, masking two failure modes: performance can collapse as length grows, and strong retrieval need not transfer to downstream use. We present ATLAS, a benchmarking framework that redefines long-context evaluation as length-dependent capability profiling. ATLAS contributes three methodological principles:(i) a layered taxonomy separating foundational operations from application workloads so failures can be attributed, (ii) length-aware AUC scoring that integrates score-length curves over a fixed 8K-1M grid, replacing single-point metrics with full degradation profiles, and (iii) ATLAScore, a harmonic-mean aggregate over taxonomy categories that penalizes imbalanced profiles, with end-to-end uncertainty propagation from subset scores through the nonlinear final aggregate. We instantiate the framework across eight capability dimensions with nine auditable components and 6,438 instances, and evaluate 26 models. Gemini-3.1-Pro-Preview leads at 128K, Claude-Opus-4.6 leads at 1M. Rankings reshuffle substantially between ATLASscore@8K-128K and ATLASscore@8K-1M: 7 models move by at least two ranks, and the two taxonomy layers share only 61% of cross-model variance, with individual rank gaps up to 12 positions. These results support reporting long-context quality by capability and length, not by a single headline score.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.28079>
- PDF: [[raw/papers/pdf/2026-atlas-all-round-testing-of-long-context-abilities-across-scales.pdf]]
- Raw markdown: [[raw/papers/md/2026-atlas-all-round-testing-of-long-context-abilities-across-scales]]
