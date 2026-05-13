---
arxiv: '2605.06216'
authors:
- Ajay Jaiswal
- Lauren Hannah
- Han-Byul Kim
- Duc Hoang
- Mehrdad Farajtabar
- Minsik Cho
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.06216
  raw: '[[raw/papers/md/2026-tide-every-layer-knows-the-token-beneath-the-context]]'
  source: https://arxiv.org/abs/2605.06216
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-tide-every-layer-knows-the-token-beneath-the-context.md
raw_pdf: raw/papers/pdf/2026-tide-every-layer-knows-the-token-beneath-the-context.pdf
read: false
slug: tide-every-layer-knows-the-token-beneath-the-context
tags:
- type/paper
- status/stub
title: 'TIDE: Every Layer Knows the Token Beneath the Context'
type: note
updated: '2026-05-11'
year: 2026
---

# TIDE: Every Layer Knows the Token Beneath the Context

> *Ajay Jaiswal, Lauren Hannah, Han-Byul Kim, Duc Hoang, Mehrdad Farajtabar, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We revisit a universally accepted but under-examined design choice in every modern LLM: a token index is looked up once at the input embedding layer and then permanently discarded. This single-injection assumption induces two structural failures: (i) the Rare Token Problem, where a Zipf-type distribution of vocabulary causes rare-token embeddings are chronically under-trained due to receiving a fraction of the cumulative gradient signal compared to common tokens; and (ii) the Contextual Collapse Problem, where limited parameters models map distributionally similar tokens to indistinguishable hidden states. As an attempt to address both, we propose TIDE, which augments the standard transformer with EmbeddingMemory: an ensemble of K independent MemoryBlocks that map token indices to context-free semantic vectors, computed once and injected into every layer through a depth-conditioned softmax router with a learnable null bank. We theoretically and empirically establish the benefits of TIDE in addressing the issues associated with single-token identity injection as well as improve performance across multiple language modeling and downstream tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.06216>
- PDF: [[raw/papers/pdf/2026-tide-every-layer-knows-the-token-beneath-the-context.pdf]]
- Raw markdown: [[raw/papers/md/2026-tide-every-layer-knows-the-token-beneath-the-context]]
