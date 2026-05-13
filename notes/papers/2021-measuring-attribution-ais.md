---
arxiv: '2112.12870'
authors:
- Hannah Rashkin
- Vitaly Nikolaev
- Matthew Lamm
- Lora Aroyo
- Michael Collins
- Dipanjan Das
- Slav Petrov
- Gaurav Singh Tomar
- Iulia Turc
- David Reitter
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2112.12870
  raw: '[[raw/papers/md/2021-measuring-attribution-ais]]'
  source: https://arxiv.org/abs/2112.12870
owner: blaz
raw_pdf: raw/papers/pdf/2021-measuring-attribution-ais.pdf
read: false
slug: measuring-attribution-ais
tags:
- type/paper
- status/stub
- domain/evals
- domain/general
title: Measuring Attribution in Natural Language Generation Models
type: note
updated: '2026-05-10'
year: 2021
---

# Measuring Attribution in Natural Language Generation Models

## Citation

- URL: https://arxiv.org/abs/2112.12870
- PDF: https://arxiv.org/pdf/2112.12870
- Authors: Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, David Reitter
- Year / venue: 2021 (arXiv, December 2021; revised 2022; published in *Transactions of the ACL*)

## Core Claim

Introduces the **Attributable to Identified Sources (AIS)** framework: a binary annotation criterion for whether a NLG model's output is fully supported by an identified source document. Validated via two-stage human annotation across conversational QA, summarisation, and table-to-text tasks.

## Key Paper Ideas

- **AIS criterion.** A model output is AIS if every claim it makes about the external world is entailed by the cited source and only that source. The criterion is deliberately conservative.
- **Two-stage annotation pipeline.** (1) Preparatory stage familiarises annotators with the source; (2) main stage judges attribution. Designed to keep inter-annotator agreement tractable.
- **Cross-task validation.** Tested on multiple NLG tasks (conv-QA, summarisation, table-to-text) to demonstrate the framework's generality.
- **Relationship to faithfulness / grounding.** AIS is a source-level grounding check, orthogonal to but complementary to fluency and factuality metrics.

## Relevance To Poolside

Useful background for content-provenance evaluation: AIS formalises what it means for a generated output to be attributable to a specific source, which is a useful conceptual anchor for synthetic-data quality checks and eval design.

## Related Notes

- Concepts: [[concepts/content-provenance-axis]]
- Maps: [[maps/evaluation/landscape]]

## Caveats

Stub — created 2026-04-28.
