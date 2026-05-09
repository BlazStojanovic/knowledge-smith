---
arxiv: '1705.07874'
authors:
- Scott Lundberg
- Su-In Lee
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2017-a-unified-approach-to-interpreting-model-predictions.md
raw_pdf: raw/papers/pdf/2017-a-unified-approach-to-interpreting-model-predictions.pdf
read: false
slug: a-unified-approach-to-interpreting-model-predictions
tags:
- interpretability
- ml
- feature-encoding
title: A Unified Approach to Interpreting Model Predictions
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1705.07874v2
venue: null
year: 2017
---

# A Unified Approach to Interpreting Model Predictions

> *Scott Lundberg, Su-In Lee* — arXiv 1705.07874, 2017

## Abstract

Understanding why a model makes a certain prediction can be as crucial as the prediction's accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability. In response, various methods have recently been proposed to help users interpret the predictions of complex models, but it is often unclear how these methods are related and when one method is preferable over another. To address this problem, we present a unified framework for interpreting predictions, SHAP (SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. Its novel components include: (1) the identification of a new class of additive feature importance measures, and (2) theoretical results showing there is a unique solution in this class with a set of desirable properties. The new class unifies six existing methods, notable because several recent methods in the class lack the proposed desirable properties. Based on insights from this unification, we present new methods that show improved computational performance and/or better consistency with human intuition than previous approaches.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-a-unified-approach-to-interpreting-model-predictions]]
- PDF (gitignored): [[raw/papers/pdf/2017-a-unified-approach-to-interpreting-model-predictions.pdf]]
- arXiv: <http://arxiv.org/abs/1705.07874v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-lundberg-shap.md` before that tree was retired.*

SHAP — now the default for tabular explanations.
