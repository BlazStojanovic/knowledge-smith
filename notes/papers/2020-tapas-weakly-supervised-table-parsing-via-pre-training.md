---
arxiv: '2004.02349'
authors:
- Jonathan Herzig
- Paweł Krzysztof Nowak
- Thomas Müller
- Francesco Piccinno
- Julian Martin Eisenschlos
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2004.02349.md
raw_pdf: raw/papers/2004.02349.pdf
read: false
slug: tapas-weakly-supervised-table-parsing-via-pre-training
tags:
- nlp
- tabular
- pretraining
- transformer
- question-answering
title: 'TAPAS: Weakly Supervised Table Parsing via Pre-training'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2004.02349v2
venue: null
year: 2020
---

# TAPAS: Weakly Supervised Table Parsing via Pre-training

> *Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller…* — arXiv 2004.02349, 2020

## Abstract

Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this paper, we present TAPAS, an approach to question answering over tables without generating logical forms. TAPAS trains from weak supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding aggregation operator to such selection. TAPAS extends BERT's architecture to encode tables as input, initializes from an effective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We experiment with three different semantic parsing datasets, and find that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but with a simpler model architecture. We additionally find that transfer learning, which is trivial in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2004.02349]]
- PDF: `raw/papers/2004.02349.pdf`
- arXiv: <http://arxiv.org/abs/2004.02349v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-herzig-tapas.md` before that tree was retired.*

Google's TAPAS — weakly-supervised table question answering.
