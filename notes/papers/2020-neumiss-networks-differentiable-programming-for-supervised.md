---
arxiv: '2007.01627'
authors:
- Marine Le Morvan
- Julie Josse
- Thomas Moreau
- Erwan Scornet
- Gaël Varoquaux
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2020-neumiss-networks-differentiable-programming-for-supervised.md
raw_pdf: raw/papers/pdf/2020-neumiss-networks-differentiable-programming-for-supervised.pdf
read: false
slug: neumiss-networks-differentiable-programming-for-supervised
tags:
- missing-data
- tabular
- ml
- generalization
title: 'NeuMiss networks: differentiable programming for supervised learning with
  missing values'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2007.01627v4
venue: null
year: 2020
---

# NeuMiss networks: differentiable programming for supervised learning with missing values

> *Marine Le Morvan, Julie Josse, Thomas Moreau…* — arXiv 2007.01627, 2020

## Abstract

The presence of missing values makes supervised learning much more challenging. Indeed, previous work has shown that even when the response is a linear function of the complete data, the optimal predictor is a complex function of the observed entries and the missingness indicator. As a result, the computational or sample complexities of consistent approaches depend on the number of missing patterns, which can be exponential in the number of dimensions. In this work, we derive the analytical form of the optimal predictor under a linearity assumption and various missing data mechanisms including Missing at Random (MAR) and self-masking (Missing Not At Random). Based on a Neumann-series approximation of the optimal predictor, we propose a new principled architecture, named NeuMiss networks. Their originality and strength come from the use of a new type of non-linearity: the multiplication by the missingness indicator. We provide an upper bound on the Bayes risk of NeuMiss networks, and show that they have good predictive accuracy with both a number of parameters and a computational complexity independent of the number of missing data patterns. As a result they scale well to problems with many features, and remain statistically efficient for medium-sized samples. Moreover, we show that, contrary to procedures using EM or imputation, they are robust to the missing data mechanism, including difficult MNAR settings such as self-masking.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2020-neumiss-networks-differentiable-programming-for-supervised]]
- PDF: `raw/papers/pdf/2020-neumiss-networks-differentiable-programming-for-supervised.pdf`
- arXiv: <http://arxiv.org/abs/2007.01627v4>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-lemorvan-neumiss.md` before that tree was retired.*

NeuMiss — missingness-aware differentiable architecture for prediction with missing data.
