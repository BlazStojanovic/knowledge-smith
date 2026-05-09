---
arxiv: '1910.03225'
authors:
- Tony Duan
- Anand Avati
- Daisy Yi Ding
- Khanh K. Thai
- Sanjay Basu
- Andrew Y. Ng
- Alejandro Schuler
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1910.03225.md
raw_pdf: raw/papers/1910.03225.pdf
read: false
slug: ngboost-natural-gradient-boosting-for-probabilistic
tags: []
title: 'NGBoost: Natural Gradient Boosting for Probabilistic Prediction'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1910.03225v4
venue: null
year: 2019
---

# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

> *Tony Duan, Anand Avati, Daisy Yi Ding…* — arXiv 1910.03225, 2019

## Abstract

We present Natural Gradient Boosting (NGBoost), an algorithm for generic probabilistic prediction via gradient boosting. Typical regression models return a point estimate, conditional on covariates, but probabilistic regression models output a full probability distribution over the outcome space, conditional on the covariates. This allows for predictive uncertainty estimation -- crucial in applications like healthcare and weather forecasting. NGBoost generalizes gradient boosting to probabilistic regression by treating the parameters of the conditional distribution as targets for a multiparameter boosting algorithm. Furthermore, we show how the Natural Gradient is required to correct the training dynamics of our multiparameter boosting approach. NGBoost can be used with any base learner, any family of distributions with continuous parameters, and any scoring rule. NGBoost matches or exceeds the performance of existing methods for probabilistic prediction while offering additional benefits in flexibility, scalability, and usability. An open-source implementation is available at github.com/stanfordmlgroup/ngboost.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1910.03225]]
- PDF: `raw/papers/1910.03225.pdf`
- arXiv: <http://arxiv.org/abs/1910.03225v4>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2020-duan-ngboost]]
