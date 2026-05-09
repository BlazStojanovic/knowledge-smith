---
arxiv: '1905.04610'
authors:
- Scott M. Lundberg
- Gabriel Erion
- Hugh Chen
- Alex DeGrave
- Jordan M. Prutkin
- Bala Nair
- Ronit Katz
- Jonathan Himmelfarb
- Nisha Bansal
- Su-In Lee
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2019-explainable-ai-for-trees-from-local-explanations-to-global.md
raw_pdf: raw/papers/pdf/2019-explainable-ai-for-trees-from-local-explanations-to-global.pdf
read: false
slug: explainable-ai-for-trees-from-local-explanations-to-global
tags:
- interpretability
- decision-tree
- gradient-boosting
- tabular
title: 'Explainable AI for Trees: From Local Explanations to Global Understanding'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1905.04610v1
venue: null
year: 2019
---

# Explainable AI for Trees: From Local Explanations to Global Understanding

> *Scott M. Lundberg, Gabriel Erion, Hugh Chen…* — arXiv 1905.04610, 2019

## Abstract

Tree-based machine learning models such as random forests, decision trees, and gradient boosted trees are the most popular non-linear predictive models used in practice today, yet comparatively little attention has been paid to explaining their predictions. Here we significantly improve the interpretability of tree-based models through three main contributions: 1) The first polynomial time algorithm to compute optimal explanations based on game theory. 2) A new type of explanation that directly measures local feature interaction effects. 3) A new set of tools for understanding global model structure based on combining many local explanations of each prediction. We apply these tools to three medical machine learning problems and show how combining many high-quality local explanations allows us to represent global structure while retaining local faithfulness to the original model. These tools enable us to i) identify high magnitude but low frequency non-linear mortality risk factors in the general US population, ii) highlight distinct population sub-groups with shared risk characteristics, iii) identify non-linear interaction effects among risk factors for chronic kidney disease, and iv) monitor a machine learning model deployed in a hospital by identifying which features are degrading the model's performance over time. Given the popularity of tree-based machine learning models, these improvements to their interpretability have implications across a broad set of domains.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2019-explainable-ai-for-trees-from-local-explanations-to-global]]
- PDF: [[raw/papers/pdf/2019-explainable-ai-for-trees-from-local-explanations-to-global.pdf]]
- arXiv: <http://arxiv.org/abs/1905.04610v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-lundberg-treeshap.md` before that tree was retired.*

TreeSHAP — polynomial-time exact Shapley values on GBDTs; the reason GBDT+SHAP became industry standard.
