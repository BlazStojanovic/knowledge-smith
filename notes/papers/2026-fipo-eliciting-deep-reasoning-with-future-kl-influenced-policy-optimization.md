---
arxiv: '2603.19835'
authors:
- Chiyu Ma
- Shuo Yang
- Kexin Huang
- Jinda Lu
- Haoming Meng
- Shangshang Wang
- Bolin Ding
- Soroush Vosoughi
- Guoyin Wang
- Jingren Zhou
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.19835
  raw: '[[raw/papers/md/2026-fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization]]'
  source: https://arxiv.org/abs/2603.19835
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization.md
raw_pdf: raw/papers/pdf/2026-fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization.pdf
read: false
slug: fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization
tags:
- type/paper
- status/stub
- reinforcement-learning
- reasoning
- credit-assignment
- llm
title: 'FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization'
type: note
updated: '2026-05-18'
year: 2026
---

# FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization

> *Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, et al.* — arXiv 2026

## TL;DR

FIPO (Future-KL Influenced Policy Optimization) targets the coarse credit assignment of GRPO-style outcome-reward (ORM) training, which spreads a single global advantage uniformly across every token in a trajectory — failing to separate critical logical pivots from trivial tokens, which the paper argues imposes a performance ceiling. FIPO folds **discounted future-KL divergence** into the policy update, yielding a dense per-token advantage that re-weights tokens by how much they influence subsequent trajectory behavior. On Qwen2.5-32B it breaks the "length stagnation" of standard baselines: average chain-of-thought grows from ~4,000 to >10,000 tokens, and AIME 2024 Pass@1 rises from 50.0% to a peak of 58.0% (~56% converged) — beating DeepSeek-R1-Zero-Math-32B (~47%) and o1-mini (~56%). Training system open-sourced, built on verl. (Summary from abstract; note unread.)

## Abstract

We present Future-KL Influenced Policy Optimization (FIPO), a reinforcement learning algorithm designed to overcome reasoning bottlenecks in large language models. While GRPO style training scales effectively, it typically relies on outcome-based rewards (ORM) that distribute a global advantage uniformly across every token in a trajectory. We argue that this coarse-grained credit assignment imposes a performance ceiling by failing to distinguish critical logical pivots from trivial tokens. FIPO addresses this by incorporating discounted future-KL divergence into the policy update, creating a dense advantage formulation that re-weights tokens based on their influence on subsequent trajectory behavior. Empirically, FIPO enables models to break through the length stagnation seen in standard baselines. Evaluated on Qwen2.5-32B, FIPO extends the average chain-of-thought length from roughly 4,000 to over 10,000 tokens and increases AIME 2024 Pass@1 accuracy from 50.0% to a peak of 58.0% (converging at approximately 56.0\%). This outperforms both DeepSeek-R1-Zero-Math-32B (around 47.0%) and o1-mini (approximately 56.0%). Our results suggest that establishing dense advantage formulations is a vital path for evolving ORM-based algorithms to unlock the full reasoning potential of base models. We open-source our training system, built on the verl framework.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.19835>
- PDF: [[raw/papers/pdf/2026-fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization.pdf]]
- Raw markdown: [[raw/papers/md/2026-fipo-eliciting-deep-reasoning-with-future-kl-influenced-policy-optimization]]
