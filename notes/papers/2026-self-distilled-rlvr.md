---
arxiv: '2604.03128'
authors:
- Chenxu Yang
- Chuanyu Qin
- Qingyi Si
- Minghui Chen
- Naibin Gu
- Dingyu Yao
- Zheng Lin
- Weiping Wang
- Jiaqi Wang
- Nan Duan
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.03128
  raw: '[[raw/papers/md/2026-self-distilled-rlvr]]'
  source: https://arxiv.org/abs/2604.03128
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-self-distilled-rlvr.md
raw_pdf: raw/papers/pdf/2026-self-distilled-rlvr.pdf
read: false
slug: self-distilled-rlvr
tags:
- type/paper
- status/stub
title: Self-Distilled RLVR
type: note
updated: '2026-05-11'
year: 2026
---

# Self-Distilled RLVR

> *Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

On-policy distillation (OPD) has become a popular training paradigm in the LLM community. This paradigm selects a larger model as the teacher to provide dense, fine-grained signals for each sampled trajectory, in contrast to reinforcement learning with verifiable rewards (RLVR), which only obtains sparse signals from verifiable outcomes in the environment. Recently, the community has explored on-policy self-distillation (OPSD), where the same model serves as both teacher and student, with the teacher receiving additional privileged information such as reference answers to enable self-evolution. This paper demonstrates that learning signals solely derived from the privileged teacher result in severe information leakage and unstable long-term training. Accordingly, we identify the optimal niche for self-distillation and propose \textbf{RLSD} (\textbf{RL}VR with \textbf{S}elf-\textbf{D}istillation). Specifically, we leverage self-distillation to obtain token-level policy differences for determining fine-grained update magnitudes, while continuing to use RLVR to derive reliable update directions from environmental feedback (e.g., response correctness). This enables RLSD to simultaneously harness the strengths of both RLVR and OPSD, achieving a higher convergence ceiling and superior training stability.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.03128>
- PDF: [[raw/papers/pdf/2026-self-distilled-rlvr.pdf]]
- Raw markdown: [[raw/papers/md/2026-self-distilled-rlvr]]
