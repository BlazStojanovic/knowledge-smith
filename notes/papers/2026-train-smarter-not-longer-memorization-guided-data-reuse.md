---
authors:
- Anonymous
created: '2026-05-12'
kind: paper
links:
  code: null
  paper: https://openreview.net/pdf?id=jTnWdO7Ld4
  raw: '[[raw/papers/md/2026-train-smarter-not-longer-memorization-guided-data-reuse]]'
  source: https://openreview.net/pdf?id=jTnWdO7Ld4
owner: blaz
parser: read
raw_md: raw/papers/md/2026-train-smarter-not-longer-memorization-guided-data-reuse.md
raw_pdf: raw/papers/pdf/2026-train-smarter-not-longer-memorization-guided-data-reuse.pdf
read: false
slug: train-smarter-not-longer-memorization-guided-data-reuse
tags:
- type/paper
- status/stub
title: 'Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM
  Training'
type: note
updated: '2026-05-12'
venue: DATA-FM workshop @ ICLR 2026
year: 2026
---

# Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training

> *Anonymous authors* — DATA-FM workshop @ ICLR 2026 (double-blind)

## TL;DR

(stub — fill in after reading)

## Abstract

The training paradigm of large language models has shifted from traditional one-pass training to multi-epoch training, as reasonable reuse of limited high-quality data can improve both model performance and sample efficiency. Meanwhile, excessive repetition introduces the risk of overfitting and diminishing returns. Determining when and how to reuse data effectively thus emerges as a natural but underexplored question. Through a novel observation of model's Memorization Window signals derived from loss retention dynamics and downstream evaluation scores, we propose Memorization-guided Data Reuse, a training paradigm that adaptively determines when and how data should be reused, enabling principled decisions on the number of training epochs and the scheduling of data replays. Our preliminary experiments reveal a consistent memorization-driven regime: performance continues to improve with repetition far beyond current practice (e.g., the commonly cited four-epoch limit). While a full scheduler remains future work, these insights provide a foundation for memorization-aware training schedules, helping to determine reuse budgets and move toward training LLMs smarter rather than longer with limited high-quality data.

## Notes

(stub)

## Source

- OpenReview: <https://openreview.net/pdf?id=jTnWdO7Ld4>
- PDF: [[raw/papers/pdf/2026-train-smarter-not-longer-memorization-guided-data-reuse.pdf]]
