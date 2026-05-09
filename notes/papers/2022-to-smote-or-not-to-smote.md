---
arxiv: '2201.08528'
authors:
- Yotam Elor
- Hadar Averbuch-Elor
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2201.08528.md
raw_pdf: raw/papers/2201.08528.pdf
read: false
slug: to-smote-or-not-to-smote
tags:
- tabular
- fairness
- imbalanced-data
- benchmark
title: To SMOTE, or not to SMOTE?
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2201.08528v3
venue: null
year: 2022
---

# To SMOTE, or not to SMOTE?

> *Yotam Elor, Hadar Averbuch-Elor* — arXiv 2201.08528, 2022

## Abstract

Balancing the data before training a classifier is a popular technique to address the challenges of imbalanced binary classification in tabular data. Balancing is commonly achieved by duplication of minority samples or by generation of synthetic minority samples. While it is well known that balancing affects each classifier differently, most prior empirical studies did not include strong state-of-the-art (SOTA) classifiers as baselines. In this work, we are interested in understanding whether balancing is beneficial, particularly in the context of SOTA classifiers. Thus, we conduct extensive experiments considering three SOTA classifiers along the weaker learners used in previous investigations. Additionally, we carefully discern proper metrics, consistent and non-consistent algorithms and hyper-parameter selection methods and show that these have a significant impact on prediction quality and on the effectiveness of balancing. Our results support the known utility of balancing for weak classifiers. However, we find that balancing does not improve prediction performance for the strong ones. We further identify several other scenarios for which balancing is effective and observe that prior studies demonstrated the utility of balancing by focusing on these settings.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2201.08528]]
- PDF: `raw/papers/2201.08528.pdf`
- arXiv: <http://arxiv.org/abs/2201.08528v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-gardner-subgroup-trees.md` before that tree was retired.*

"Subgroup Robustness Grows on Trees" — GBDTs tend to be more robust out of the box.
