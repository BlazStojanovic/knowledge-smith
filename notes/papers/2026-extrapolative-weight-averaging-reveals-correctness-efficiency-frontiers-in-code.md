---
arxiv: '2605.28751'
authors:
- Kunhao Zheng
- Pierre Chambon
- Juliette Decugis
- Jonas Gehring
- Taco Cohen
- Benjamin Negrevergne
- Gabriel Synnaeve
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.28751
  raw: '[[raw/papers/md/2026-extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code]]'
  source: https://arxiv.org/abs/2605.28751
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code.md
raw_pdf: raw/papers/pdf/2026-extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code.pdf
read: false
slug: extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code
tags:
- type/paper
- status/stub
title: Extrapolative Weight Averaging Reveals Correctness-Efficiency Frontiers in
  Code RL
type: note
updated: '2026-06-01'
year: 2026
---

# Extrapolative Weight Averaging Reveals Correctness-Efficiency Frontiers in Code RL

> *Kunhao Zheng, Pierre Chambon, Juliette Decugis, Jonas Gehring, Taco Cohen, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Linear interpolation between fine-tuned checkpoints has been shown to trace the Pareto front between competing objectives, but whether extrapolative weight averaging can extend such frontiers to new checkpoints useful at inference time, without additional RL training, remains unclear. We study this question in RL for competitive programming, where hidden unit tests under time and memory limits enforce both functional correctness and computational efficiency. Starting from a shared initialization, we train checkpoints under nested unit-test coverage: low-coverage rewards require passing smaller-input tests, while high-coverage rewards require passing progressively larger tests up to the full suite. This sweep reveals the emergence of a correctness-efficiency frontier: on hard problems, higher-coverage reward reduces optimization failures but increases correctness failures, leaving solve rate nearly unchanged. Interpolation between low- and high-coverage checkpoints recovers this frontier, while extrapolation extends it beyond the trained endpoints. Both the frontier and its extrapolative continuation appear across three inference settings, pure reasoning, tool use, and agentic coding, and across two model scales, 32B and 7B. At the problem level, moving along the frontier changes which problems are solved, making extrapolated checkpoints complementary policies in inference-time scaling. Ensembles with extrapolative weight averaging broaden coverage and improve pass@250 on LCB/hard by 3.3% over the best single checkpoint at matched sample budget. These results show that nested unit-test coverage in code RL induces a frontier that extrapolative weight averaging can navigate, extend, and exploit.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.28751>
- PDF: [[raw/papers/pdf/2026-extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code.pdf]]
- Raw markdown: [[raw/papers/md/2026-extrapolative-weight-averaging-reveals-correctness-efficiency-frontiers-in-code]]
