---
arxiv: '2605.05365'
authors:
- Robert Washbourne
- Rishi Iyer
- Tomas Figliolia
- Henry Zheng
- Ryan Lorig-Roach
- Sungyeon Yang
- Pritish Yuvraj
- Quentin Anthony
- Yury Tokpanov
- Xiao Yang
- Ganesh Nanduru
- Stephen Ebert
- Praneeth Medepalli
- Skyler Szot
- Srivatsan Rajagopal
- Alex Ong
- Bhavana Mehta
- Beren Millidge
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.05365
  raw: '[[raw/papers/md/2026-zaya1-8b-technical-report]]'
  source: https://arxiv.org/abs/2605.05365
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-zaya1-8b-technical-report.md
raw_pdf: raw/papers/pdf/2026-zaya1-8b-technical-report.pdf
read: false
slug: zaya1-8b-technical-report
tags:
- type/paper
- status/stub
title: ZAYA1-8B Technical Report
type: note
updated: '2026-05-11'
year: 2026
---

# ZAYA1-8B Technical Report

> *Robert Washbourne, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We present ZAYA1-8B, a reasoning-focused mixture-of-experts (MoE) model with 700M active and 8B total parameters, built on Zyphra's MoE++ architecture. ZAYA1-8B's core pretraining, midtraining, and supervised fine-tuning (SFT) were performed on a full-stack AMD compute, networking, and software platform. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, and remains competitive with substantially larger open-weight reasoning models. ZAYA1-8B was trained from scratch for reasoning, with reasoning data included from pretraining onward using an answer-preserving trimming scheme. Post-training uses a four-stage RL cascade: reasoning warmup on math and puzzles; a 400-task RLVE-Gym curriculum; math and code RL with test-time compute traces and synthetic code environments built from competitive-programming references; and behavioral RL for chat and instruction following. We also introduce Markovian RSA, a test-time compute method that recursively aggregates parallel reasoning traces while carrying forward only bounded-length reasoning tails between rounds. In TTC evaluation, Markovian RSA raises ZAYA1-8B to 91.9\% on AIME'25 and 89.6\% on HMMT'25 while carrying forward only a 4K-token tail, narrowing the gap to much larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, and GPT-5-High.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.05365>
- PDF: [[raw/papers/pdf/2026-zaya1-8b-technical-report.pdf]]
- Raw markdown: [[raw/papers/md/2026-zaya1-8b-technical-report]]
