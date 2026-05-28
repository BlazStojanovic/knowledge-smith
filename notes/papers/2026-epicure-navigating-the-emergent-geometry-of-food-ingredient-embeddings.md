---
arxiv: '2605.22391'
authors:
- Jakub Radzikowski
- Josef Chen
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.22391
  raw: '[[raw/papers/md/2026-epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings]]'
  source: https://arxiv.org/abs/2605.22391
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings.md
raw_pdf: raw/papers/pdf/2026-epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings.pdf
read: false
slug: epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings
tags:
- type/paper
- status/stub
title: 'Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings'
type: note
updated: '2026-05-28'
year: 2026
---

# Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings

> *Jakub Radzikowski, Josef Chen* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We present Epicure, a family of three sibling skip-gram ingredient embeddings retrained from scratch on a multilingual recipe corpus. We aggregate 4.14M recipes from 11 sources spanning seven languages, English, Chinese, Russian, Vietnamese, Spanish, Turkish, Indonesian, German, and Indian-English, and normalise the raw ingredient strings to 1,790 canonical entries via an LLM-augmented pipeline. A 203,508-edge ingredient-ingredient NPMI graph and an 80,019-edge typed FlavorDB ingredient-compound graph, 2,247 typed compound nodes across 15 categories, seed three Metapath2Vec variants that share architecture and hyperparameters and differ only in the random-walk schema: Cooc walks the co-occurrence graph only, Chem walks the typed compound metapaths only, and Core blends both via injected ingredient-ingredient walks at controlled mixing, placing each model at a distinct point on the chemistry-vs-recipe-context spectrum.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.22391>
- PDF: [[raw/papers/pdf/2026-epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings.pdf]]
- Raw markdown: [[raw/papers/md/2026-epicure-navigating-the-emergent-geometry-of-food-ingredient-embeddings]]
