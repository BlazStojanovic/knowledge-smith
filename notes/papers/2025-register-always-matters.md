---
aliases:
- Register Always Matters
arxiv: '2504.01542'
authors:
- Amanda Myntti
- Erik Henriksson
- Veronika Laippala
- Sampo Pyysalo (University of Turku)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.01542
  raw: '[[raw/papers/md/2025-register-always-matters]]'
  source: https://arxiv.org/abs/2504.01542
owner: blaz
raw_pdf: raw/papers/pdf/2025-register-always-matters.pdf
read: false
slug: register-always-matters
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'Register Always Matters: Analysis of LLM Pretraining Data Through the Lens
  of Language Variation'
type: note
updated: '2026-05-10'
year: 2025
---

# Register Always Matters: Analysis of LLM Pretraining Data Through the Lens of Language Variation

## Citation

- URL: https://arxiv.org/abs/2504.01542
- PDF: https://arxiv.org/pdf/2504.01542
- Authors: Amanda Myntti, Erik Henriksson, Veronika Laippala, Sampo Pyysalo (University of Turku)
- Year / venue: 2025-04 arXiv preprint; COLM 2025
- Raw PDF: [[raw/papers/pdf/2025-register-always-matters.pdf]]

## Core Claim

The register (genre) of pretraining data substantially affects LLM performance independently of quality scores, and selective register-based data composition can improve or harm model capabilities — demonstrating that text type is an informative axis orthogonal to quality for data curation.

## Key Paper Ideas

- **First register-based pretraining study**: uses the CORE register scheme (9 main registers, 25 subregisters in a hierarchy) to curate pretraining data and measure per-register downstream performance.
- **Register scheme**: How-to-Instructions (HI), Interactive Discussion (ID), Informational Description (IN), Informational Persuasion (IP), Lyrical (LY), Machine Translation (MT), Narrative (NA), Opinion (OP), Spoken (SP), plus subregisters News (ne) and Description (dtp).
- **Register ≠ quality**: selecting by register captures a fundamentally different dimension than quality scoring. A "high-quality" News document and a "high-quality" Opinion blog train models differently.
- **Categorical, not ordinal**: registers are types, not a ranked scale. There is no "best" register — only registers appropriate for specific downstream capabilities.

## Methodology

- Source data: HPLT v2 deduplicated corpus with register labels from XLM-RoBERTa-Large classifier (Henriksson et al. 2024).
- Train 1.71B parameter LLaMA-architecture models on 100B tokens per register class.
- Evaluate on 10 zero-shot benchmarks (HellaSwag, WinoGrande, PIQA, SIQA, OpenBookQA, ARC, CommonsenseQA, MMLU).
- Compare individual registers, register combinations, and unfiltered baselines.

## Key Results

- **News register → subpar performance**: despite being common in pretraining data, News alone underperforms.
- **Opinion → highly beneficial**: reviews, opinion blogs boost model performance substantially.
- **Combining well-performing registers**: How-to-Instructions + Informational Description + Opinion together → major improvements over single-register models.
- **Unfiltered outperforms single-register**: a model trained on the full distribution of registers outperforms any single-register model, but curated combinations can beat unfiltered.
- **Per-benchmark register effects**: How-to-Instructions excels at physical reasoning/sentence completion but barely crosses random on world-knowledge benchmarks. Narrative boosts social interaction but struggles with scientific questions. Register effects are task-specific.

## Core Concepts

- Existing concepts: [[concepts/data-filtering-paradigms]], [[concepts/multi-property-data-curation]]

## Relevance To Poolside

*Our interpretation.* Register provides a mode-collapse diagnostic for synthetic data that quality scores miss. If a generator produces text that scores well on quality but collapses register diversity (e.g., all outputs are informational/expository), register classification can detect this. For code domains, the analogue might be "code register" — tutorial code vs. production code vs. test code vs. configuration, each training different capabilities.

## Key Follow-Ups / Jumping-Off Points

- [[notes/papers/2025-hplt-3-very-large-scale-multilingual-resources]] — applies register annotations at web scale alongside quality and PII labels.
- Connection to [[concepts/diversity]] — register distribution is a meso-level ($\beta$) diversity axis. Register collapse = $\beta$-diversity loss.

## Related Notes

- Concepts: [[concepts/data-filtering-paradigms]], [[concepts/multi-property-data-curation]], [[concepts/diversity]]
- Maps: [[maps/evaluation/point-level]]

## Caveats

- English only. Register classification may behave differently in other languages.
- 1.71B parameter scale only; register effects at frontier model scale unvalidated.
- CORE register scheme is one taxonomy among many; choice of register taxonomy may affect conclusions.
- Register labels are classifier-predicted (not human-annotated), introducing noise.
