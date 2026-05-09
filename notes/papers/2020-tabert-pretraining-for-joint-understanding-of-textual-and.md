---
arxiv: '2005.08314'
authors:
- Pengcheng Yin
- Graham Neubig
- Wen-tau Yih
- Sebastian Riedel
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2020-tabert-pretraining-for-joint-understanding-of-textual-and.md
raw_pdf: raw/papers/pdf/2020-tabert-pretraining-for-joint-understanding-of-textual-and.pdf
read: false
slug: tabert-pretraining-for-joint-understanding-of-textual-and
tags:
- pretraining
- tabular
- nlp
- transformer
title: 'TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2005.08314v1
venue: null
year: 2020
---

# TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data

> *Pengcheng Yin, Graham Neubig, Wen-tau Yih…* — arXiv 2005.08314, 2020

## Abstract

Recent years have witnessed the burgeoning of pretrained language models (LMs) for text-based natural language (NL) understanding tasks. Such models are typically trained on free-form NL text, hence may not be suitable for tasks like semantic parsing over structured data, which require reasoning over both free-form NL questions and structured tabular data (e.g., database tables). In this paper we present TaBERT, a pretrained LM that jointly learns representations for NL sentences and (semi-)structured tables. TaBERT is trained on a large corpus of 26 million tables and their English contexts. In experiments, neural semantic parsers using TaBERT as feature representation layers achieve new best results on the challenging weakly-supervised semantic parsing benchmark WikiTableQuestions, while performing competitively on the text-to-SQL dataset Spider. Implementation of the model will be available at http://fburl.com/TaBERT .

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2020-tabert-pretraining-for-joint-understanding-of-textual-and]]
- PDF: `raw/papers/pdf/2020-tabert-pretraining-for-joint-understanding-of-textual-and.pdf`
- arXiv: <http://arxiv.org/abs/2005.08314v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-yin-tabert.md` before that tree was retired.*

TaBERT — BERT applied to natural language plus linearized tables (semantic, not predictive).
