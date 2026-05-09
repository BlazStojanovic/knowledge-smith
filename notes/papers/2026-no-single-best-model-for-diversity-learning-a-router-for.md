---
arxiv: '2604.02319'
authors:
- Yuhan Liu
- Fangyuan Xu
- Vishakh Padmakumar
- Daphne Ippolito
- Eunsol Choi
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2026-no-single-best-model-for-diversity-learning-a-router-for.md
raw_pdf: raw/papers/pdf/2026-no-single-best-model-for-diversity-learning-a-router-for.pdf
read: false
slug: no-single-best-model-for-diversity-learning-a-router-for
tags:
- mixture-of-experts
- llm
- evaluation
title: 'No Single Best Model for Diversity: Learning a Router for Sample Diversity'
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2604.02319
venue: null
year: 2026
---

# No Single Best Model for Diversity: Learning a Router for Sample Diversity

> *Yuhan Liu, Fangyuan Xu, Vishakh Padmakumar…* — arXiv 2604.02319, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

When posed with prompts that permit a large number of valid answers, comprehensively generating them is the first step towards satisfying a wide range of users. In this paper, we study methods to elicit a comprehensive set of valid responses. To evaluate this, we introduce diversity coverage, a metric that measures the total quality scores assigned to each unique answer in the predicted answer set relative to the best possible answer set with the same number of answers. Using this metric, we evaluate 18 LLMs, finding no single model dominates at generating diverse responses to a wide range of open-ended prompts. Yet, per each prompt, there exists a model that outperforms all other models significantly at generating a diverse answer set. Motivated by this finding, we introduce a router that predicts the best model for each query. On NB-Wildchat, our trained router outperforms the single best model baseline (26.3% vs 23.8%). We further show generalization to an out-of-domain dataset (NB-Curated) as well as different answer-generation prompting strategies. Our work lays foundation for studying generating comprehensive answers when we have access to a suite of models.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-no-single-best-model-for-diversity-learning-a-router-for]]
- PDF: [[raw/papers/pdf/2026-no-single-best-model-for-diversity-learning-a-router-for.pdf]]
- arXiv: <https://arxiv.org/abs/2604.02319>
