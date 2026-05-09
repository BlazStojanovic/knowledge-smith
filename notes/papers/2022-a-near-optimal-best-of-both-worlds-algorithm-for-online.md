---
arxiv: '2206.00557'
authors:
- Chloé Rouyer
- Dirk van der Hoeven
- Nicolò Cesa-Bianchi
- Yevgeny Seldin
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2206.00557.md
raw_pdf: raw/papers/2206.00557.pdf
read: false
slug: a-near-optimal-best-of-both-worlds-algorithm-for-online
tags:
- optimization
- theory
- rl
title: A Near-Optimal Best-of-Both-Worlds Algorithm for Online Learning with Feedback
  Graphs
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2206.00557v1
venue: null
year: 2022
---

# A Near-Optimal Best-of-Both-Worlds Algorithm for Online Learning with Feedback Graphs

> *Chloé Rouyer, Dirk van der Hoeven, Nicolò Cesa-Bianchi…* — arXiv 2206.00557, 2022

## Abstract

We consider online learning with feedback graphs, a sequential decision-making framework where the learner's feedback is determined by a directed graph over the action set. We present a computationally efficient algorithm for learning in this framework that simultaneously achieves near-optimal regret bounds in both stochastic and adversarial environments. The bound against oblivious adversaries is $\tilde{O} (\sqrt{αT})$, where $T$ is the time horizon and $α$ is the independence number of the feedback graph. The bound against stochastic environments is $O\big( (\ln T)^2 \max_{S\in \mathcal I(G)} \sum_{i \in S} Δ_i^{-1}\big)$ where $\mathcal I(G)$ is the family of all independent sets in a suitably defined undirected version of the graph and $Δ_i$ are the suboptimality gaps. The algorithm combines ideas from the EXP3++ algorithm for stochastic and adversarial bandits and the EXP3.G algorithm for feedback graphs with a novel exploration scheme. The scheme, which exploits the structure of the graph to reduce exploration, is key to obtain best-of-both-worlds guarantees with feedback graphs. We also extend our algorithm and results to a setting where the feedback graphs are allowed to change over time.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2206.00557]]
- PDF: `raw/papers/2206.00557.pdf`
- arXiv: <http://arxiv.org/abs/2206.00557v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-schafl-hopular.md` before that tree was retired.*

Hopular — modern Hopfield layers applied to tabular data.
