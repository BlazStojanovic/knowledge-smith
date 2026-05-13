---
arxiv: '2205.10487'
authors:
- Danny Hernandez
- Tom Brown
- Tom Conerly
- Nova DasSarma
- Dawn Drain
- Sheer El-Showk
- Nelson Elhage
- Zac Hatfield-Dodds
- Tom Henighan
- Tristan Hume
- Scott Johnston
- Ben Mann
- Chris Olah
- Catherine Olsson
- Dario Amodei
- Nicholas Joseph
- Jared Kaplan
- Sam McCandlish
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2205.10487
  raw: '[[raw/papers/md/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data]]'
  source: https://arxiv.org/abs/2205.10487
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data.md
raw_pdf: raw/papers/pdf/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data.pdf
read: false
slug: scaling-laws-and-interpretability-of-learning-from-repeated-data
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/repetition
- domain/data-mix
- domain/models
title: Scaling Laws and Interpretability of Learning from Repeated Data
type: note
updated: '2026-05-11'
year: 2022
---

# Scaling Laws and Interpretability of Learning from Repeated Data

## Citation

- URL: https://arxiv.org/abs/2205.10487
- PDF: https://arxiv.org/pdf/2205.10487
- Authors: Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, et al. (18 authors)
- Year / venue: 2022-05-21 arXiv preprint
- arXiv: 2205.10487v1
- Categories: cs.LG, cs.AI
- Raw PDF: [[raw/papers/pdf/2022-scaling-laws-and-interpretability-of-learning-from-repeated-data.pdf]]
- Source filename: `2205.10487v1.pdf`

## Short Summary

Recent large language models have been trained on vast datasets, but also often on repeated data, either intentionally for the purpose of upweighting higher quality data, or unintentionally because data deduplication is not perfect and the model is exposed to repeated data at the sentence, paragraph, or document level. Some works have reported substantial negative performance effects of this repeated data.

## Relevance To Poolside

Our interpretation: keep this as an unread source for future grounding. Use it when its method or claim becomes load-bearing for a Poolside hypothesis, experiment, model note, or data-method decision.

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Repetition mode: partial repetition, where a small fraction of the dataset is repeated many times inside an otherwise unique corpus.
- Memo-grounded claim: this regime can be much more damaging than full-dataset epochs; the memo records a double-descent-in-epochs effect and disproportionate harm around 3-10% repeated data.
- Poolside implication: be careful with up-weighting small high-quality subsets; localized repetition may not behave like uniform epoch-level repetition.

## Related Notes

- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
- [[concepts/scaling-laws-foundational]] — data-constrained extension
- [[concepts/data-repetition]] — partial vs full-dataset repetition regimes
- [[maps/scaling-laws/landscape]] — data-constrained domain

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
