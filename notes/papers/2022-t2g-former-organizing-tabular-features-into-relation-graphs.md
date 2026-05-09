---
arxiv: '2211.16887'
authors:
- Jiahuan Yan
- Jintai Chen
- Yixuan Wu
- Danny Z. Chen
- Jian Wu
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2211.16887.md
raw_pdf: raw/papers/2211.16887.pdf
read: false
slug: t2g-former-organizing-tabular-features-into-relation-graphs
tags: []
title: 'T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous
  Feature Interaction'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2211.16887v2
venue: null
year: 2022
---

# T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction

> *Jiahuan Yan, Jintai Chen, Yixuan Wu…* — arXiv 2211.16887, 2022

## Abstract

Recent development of deep neural networks (DNNs) for tabular learning has largely benefited from the capability of DNNs for automatic feature interaction. However, the heterogeneity nature of tabular features makes such features relatively independent, and developing effective methods to promote tabular feature interaction still remains an open problem. In this paper, we propose a novel Graph Estimator, which automatically estimates the relations among tabular features and builds graphs by assigning edges between related features. Such relation graphs organize independent tabular features into a kind of graph data such that interaction of nodes (tabular features) can be conducted in an orderly fashion. Based on our proposed Graph Estimator, we present a bespoke Transformer network tailored for tabular learning, called T2G-Former, which processes tabular data by performing tabular feature interaction guided by the relation graphs. A specific Cross-level Readout collects salient features predicted by the layers in T2G-Former across different levels, and attains global semantics for final prediction. Comprehensive experiments show that our T2G-Former achieves superior performance among DNNs and is competitive with non-deep Gradient Boosted Decision Tree models.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2211.16887]]
- PDF: `raw/papers/2211.16887.pdf`
- arXiv: <http://arxiv.org/abs/2211.16887v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2023-yan-t2gformer]]
