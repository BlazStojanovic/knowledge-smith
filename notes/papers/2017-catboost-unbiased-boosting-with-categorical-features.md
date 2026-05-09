---
arxiv: '1706.09516'
authors:
- Liudmila Prokhorenkova
- Gleb Gusev
- Aleksandr Vorobev
- Anna Veronika Dorogush
- Andrey Gulin
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2017-catboost-unbiased-boosting-with-categorical-features.md
raw_pdf: raw/papers/pdf/2017-catboost-unbiased-boosting-with-categorical-features.pdf
read: false
slug: catboost-unbiased-boosting-with-categorical-features
tags:
- gradient-boosting
- tabular
- feature-encoding
- decision-tree
title: 'CatBoost: unbiased boosting with categorical features'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1706.09516v5
venue: null
year: 2017
---

# CatBoost: unbiased boosting with categorical features

> *Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev…* — arXiv 1706.09516, 2017

## Abstract

This paper presents the key algorithmic techniques behind CatBoost, a new gradient boosting toolkit. Their combination leads to CatBoost outperforming other publicly available boosting implementations in terms of quality on a variety of datasets. Two critical algorithmic advances introduced in CatBoost are the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features. Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing implementations of gradient boosting algorithms. In this paper, we provide a detailed analysis of this problem and demonstrate that proposed algorithms solve it effectively, leading to excellent empirical results.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-catboost-unbiased-boosting-with-categorical-features]]
- PDF: `raw/papers/pdf/2017-catboost-unbiased-boosting-with-categorical-features.pdf`
- arXiv: <http://arxiv.org/abs/1706.09516v5>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2018-prokhorenkova-catboost.md` before that tree was retired.*

CatBoost — ordered boosting with native categorical feature handling.
