---
arxiv: '1412.6980'
authors:
- Diederik P. Kingma
- Jimmy Ba
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1412.6980.md
raw_pdf: raw/papers/1412.6980.pdf
read: false
slug: adam-a-method-for-stochastic-optimization
tags:
- optimization
title: 'Adam: A Method for Stochastic Optimization'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1412.6980v9
venue: null
year: 2014
---

# Adam: A Method for Stochastic Optimization

> *Diederik P. Kingma, Jimmy Ba* — arXiv 1412.6980, 2014

## Abstract

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1412.6980]]
- PDF (gitignored): `raw/papers/1412.6980.pdf`
- arXiv: <http://arxiv.org/abs/1412.6980v9>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2015-kingma-adam.md` before that tree was retired.*

Adam optimizer, used in every tabular-DL attempt.
