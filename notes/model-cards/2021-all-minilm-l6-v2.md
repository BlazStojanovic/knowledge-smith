---
created: '2026-05-11'
developer: Sentence Transformers
family: MiniLM
kind: model-card
license: Apache-2.0
links:
  code: null
  paper: null
  raw: null
  source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model_type: llm
owner: blaz
parameters_total: 22.7M
read: false
slug: all-minilm-l6-v2
tags:
- type/model-card
- status/stub
- domain/models
title: all-MiniLM-L6-v2
type: note
updated: '2026-05-11'
variants:
- all-MiniLM-L6-v2
year: 2021
---

# all-MiniLM-L6-v2

> *Sentence Transformers (UKP / Reimers)* — released circa 2021

## Overview

Sentence-embedding model that maps sentences/short paragraphs to 384-d dense vectors. Trained via contrastive learning on >1B sentence pairs across 28+ datasets (Reddit, S2ORC, WikiAnswers, PAQ, Stack Exchange, MS MARCO, etc.). Widely used baseline for semantic search, clustering, and retrieval; tiny enough for CPU inference.

## Model Family

| Property | Value |
|---|---|
| Developer    | Sentence Transformers (UKP-TUDA) |
| Release date | ~2021 |
| Family       | MiniLM (distilled BERT) |
| Variants     | base: nreimers/MiniLM-L6-H384-uncased |
| License      | Apache 2.0 |

## Architecture

| Property              | Value           |
|---|---|
| Parameters (total)    | 22.7M |
| Output dim            | 384 |
| Layers                | 6 |
| Max sequence length   | 128 tokens (256 word-piece truncation default) |
| Architecture          | distilled BERT (MiniLM-L6-H384-uncased) |

## Training

- Objective: contrastive (1B+ sentence pairs)
- Hardware: 7× TPU v3-8

## Reported Evals

| Eval | Score | Source |
|---|---|---|
| (see SBERT eval tables) | — | sentence-transformers docs |

## Caveats

- Release year not explicitly stated on card; ~2021 inferred from MiniLM paper + sentence-transformers history.

## Source

- HuggingFace: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
