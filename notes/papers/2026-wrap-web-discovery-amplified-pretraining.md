---
arxiv: '2604.06829'
authors:
- Jiang Zhou
- Yunhao Wang
- Xing Wu
- Tinghao Yu
- Feng Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.06829
  raw: '[[raw/papers/md/2026-wrap-web-discovery-amplified-pretraining]]'
  source: https://arxiv.org/abs/2604.06829
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-wrap-web-discovery-amplified-pretraining.md
raw_pdf: raw/papers/pdf/2026-wrap-web-discovery-amplified-pretraining.pdf
read: false
slug: wrap-web-discovery-amplified-pretraining
tags:
- type/paper
- status/stub
title: 'WRAP++: Web discoveRy Amplified Pretraining'
type: note
updated: '2026-05-11'
year: 2026
---

# WRAP++: Web discoveRy Amplified Pretraining

> *Jiang Zhou, Yunhao Wang, Xing Wu, Tinghao Yu, Feng Zhang* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Synthetic data rephrasing has emerged as a powerful technique for enhancing knowledge acquisition during large language model (LLM) pretraining. However, existing approaches operate at the single-document level, rewriting individual web pages in isolation. This confines synthesized examples to intra-document knowledge, missing cross-document relationships and leaving facts with limited associative context. We propose WRAP++ (Web discoveRy Amplified Pretraining), which amplifies the associative context of factual knowledge by discovering cross-document relationships from web hyperlinks and synthesizing joint QA over each discovered document pair. Concretely, WRAP++ discovers high-confidence relational motifs including dual-links and co-mentions, and synthesizes QA that requires reasoning across both documents. This produces relational knowledge absent from either source document alone, creating diverse entry points to the same facts. Because the number of valid entity pairs grows combinatorially, this discovery-driven synthesis also amplifies data scale far beyond single-document rewriting. Instantiating WRAP++ on Wikipedia, we amplify ~8.4B tokens of raw text into 80B tokens of cross-document QA data. On SimpleQA, OLMo-based models at both 7B and 32B scales trained with WRAP++ substantially outperform single-document approaches and exhibit sustained scaling gains, underscoring the advantage of cross-document knowledge discovery and amplification.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.06829>
- PDF: [[raw/papers/pdf/2026-wrap-web-discovery-amplified-pretraining.pdf]]
- Raw markdown: [[raw/papers/md/2026-wrap-web-discovery-amplified-pretraining]]
