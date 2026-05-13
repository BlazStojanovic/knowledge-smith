---
aliases:
- FastText
- fastText classification
arxiv: '1607.01759'
authors:
- Armand Joulin
- Edouard Grave
- Piotr Bojanowski
- Tomas Mikolov (Facebook AI Research)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1607.01759
  raw: https://arxiv.org/pdf/1607.01759
  source: https://arxiv.org/abs/1607.01759
owner: blaz
read: false
slug: bag-of-tricks-for-efficient-text-classification
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/llm
title: Bag of Tricks for Efficient Text Classification
type: note
updated: '2026-05-10'
year: 2016
---

# Bag of Tricks for Efficient Text Classification

## Citation

- URL: https://arxiv.org/abs/1607.01759
- PDF: https://arxiv.org/pdf/1607.01759
- Authors: Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov (Facebook AI Research)
- Year / venue: 2016-07 arXiv; EACL 2017

## Short Summary

Introduces fastText, a simple and efficient text classifier using bag of n-grams with a linear model. On par with deep learning classifiers in accuracy but orders of magnitude faster — trains on 1B+ words in under 10 minutes on a standard CPU; classifies 500K sentences in under a minute. Foundational tool for data curation pipelines: fastText classifiers are used by DCLM, FineWeb-2-HQ, CCNet (language ID), and many pretraining data pipelines as cheap quality/domain/language classifiers.
