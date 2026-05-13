---
arxiv: '2211.09085'
authors:
- Ross Taylor
- Marcin Kardas
- Guillem Cucurull
- Thomas Scialom
- Anthony Hartshorn
- Elvis Saravia
- Andrew Poulton
- Viktor Kerkez
- et al
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2211.09085
  raw: '[[raw/papers/md/2022-galactica-a-large-language-model-for-science]]'
  source: https://arxiv.org/abs/2211.09085
owner: blaz
raw_pdf: raw/papers/pdf/2022-galactica-a-large-language-model-for-science.pdf
read: false
slug: galactica-a-large-language-model-for-science
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/repetition
- domain/models
title: 'Galactica: A Large Language Model for Science'
type: note
updated: '2026-05-10'
year: 2022
---

# Galactica: A Large Language Model for Science

## Citation

- URL: https://arxiv.org/abs/2211.09085
- PDF: https://arxiv.org/pdf/2211.09085
- Authors: Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, et al.
- Year / venue: 2022-11-16 arXiv preprint
- arXiv: 2211.09085v1
- Raw PDF: [[raw/papers/pdf/2022-galactica-a-large-language-model-for-science.pdf]]

## Short Summary

Introduces Galactica, a large language model trained on scientific papers, reference material, knowledge bases, and related scientific corpora. In the repetition memo it is important mainly as a concrete data-constrained case study: 120B parameters, 450B tokens, and roughly 4.25 epochs over a 106B-token unique corpus.

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Role in the memo: case study used by Muennighoff et al. to compare Galactica's actual training allocation against a data-constrained compute-optimal allocation.
- Memo-grounded claim: Muennighoff et al.'s fit suggests a smaller model trained for more repeated tokens would have been preferable under the same unique-token budget.

## Relevance To Poolside

Our interpretation: useful as a cautionary example that repeated-data decisions are entangled with model-size allocation; the repeated-epoch count alone is not the whole experimental design.

## Related Notes

- [[notes/papers/2023-scaling-data-constrained-language-models]]
- [[notes/papers/2022-training-compute-optimal-large-language-models]]
- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
