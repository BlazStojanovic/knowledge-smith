---
arxiv: '2601.22950'
authors:
- Petar Veličković
- Federico Barbero
- Christos Perivolaropoulos
- Simon Osindero
- Razvan Pascanu
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.22950
  raw: '[[raw/papers/md/2026-perplexity-cannot-always-tell-right-from-wrong]]'
  source: https://arxiv.org/abs/2601.22950
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-perplexity-cannot-always-tell-right-from-wrong.md
raw_pdf: raw/papers/pdf/2026-perplexity-cannot-always-tell-right-from-wrong.pdf
read: false
slug: perplexity-cannot-always-tell-right-from-wrong
tags:
- type/paper
- llm
- evaluation
- calibration
- status/stub
title: Perplexity Cannot Always Tell Right from Wrong
type: note
updated: '2026-05-09'
venue: null
year: 2026
---

# Perplexity Cannot Always Tell Right from Wrong

> *Petar Veličković, Federico Barbero, Christos Perivolaropoulos…* — arXiv 2601.22950, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Perplexity -- a function measuring a model's overall level of "surprise" when encountering a particular output -- has gained significant traction in recent years, both as a loss function and as a simple-to-compute metric of model quality. Prior studies have pointed out several limitations of perplexity, often from an empirical manner. Here we leverage recent results on Transformer continuity to show in a rigorous manner how perplexity may be an unsuitable metric for model selection. Specifically, we prove that, if there is any sequence that a compact decoder-only Transformer model predicts accurately and confidently -- a necessary pre-requisite for strong generalisation -- it must imply existence of another sequence with very low perplexity, but not predicted correctly by that same model. Further, by analytically studying iso-perplexity plots, we find that perplexity will not always select for the more accurate model -- rather, any increase in model confidence must be accompanied by a commensurate rise in accuracy for the new model to be selected.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-perplexity-cannot-always-tell-right-from-wrong]]
- PDF: [[raw/papers/pdf/2026-perplexity-cannot-always-tell-right-from-wrong.pdf]]
- arXiv: <https://arxiv.org/abs/2601.22950>
