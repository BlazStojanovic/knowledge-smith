---
arxiv: '2603.18444'
authors:
- Haechan Kim
- Soohyun Ryu
- Gyouk Chu
- Doohyuk Jang
- Eunho Yang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.18444
  raw: '[[raw/papers/md/2026-discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement]]'
  source: https://arxiv.org/abs/2603.18444
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement.md
raw_pdf: raw/papers/pdf/2026-discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement.pdf
read: false
slug: discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement
tags:
- type/paper
- status/stub
title: Discounted Beta--Bernoulli Reward Estimation for Sample-Efficient Reinforcement
  Learning with Verifiable Rewards
type: note
updated: '2026-05-11'
year: 2026
---

# Discounted Beta--Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards

> *Haechan Kim, Soohyun Ryu, Gyouk Chu, Doohyuk Jang, Eunho Yang* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for improving the reasoning capabilities of large language models. However, existing group-based RLVR methods often suffer from severe sample inefficiency. This inefficiency stems from reliance on point estimation of rewards from a small number of rollouts, leading to high estimation variance, variance collapse, and ineffective utilization of generated responses. In this work, we reformulate RLVR from a statistical estimation perspective by modeling rewards as samples drawn from a policy-induced distribution and casting advantage computation as the problem of estimating the reward distribution from finite data. Building on this view, we propose Discounted Beta--Bernoulli (DBB) reward estimation, which leverages historical reward statistics for the non-stationary distribution. Although biased, the resulting estimator exhibits reduced and stable variance, theoretically avoids estimated variance collapse, and achieves lower mean squared error than standard point estimation. Extensive experiments across six in-distribution and three out-of-distribution reasoning benchmarks demonstrate that GRPO with DBB consistently outperforms naive GRPO, achieving average Acc@8 improvements of 3.22/2.42 points in-distribution and 12.49/6.92 points out-of-distribution on the 1.7B and 8B models, respectively, without additional computational cost or memory usage.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.18444>
- PDF: [[raw/papers/pdf/2026-discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement.pdf]]
- Raw markdown: [[raw/papers/md/2026-discounted-beta-bernoulli-reward-estimation-for-sample-efficient-reinforcement]]
