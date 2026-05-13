---
authors:
- Anthropic (Alignment Science)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://alignment.anthropic.com/2025/pretraining-data-filtering/
  raw: null
  source: https://alignment.anthropic.com/2025/pretraining-data-filtering/
owner: blaz
read: false
slug: anthropic-pretraining-data-filtering
tags:
- type/paper
- source/engineering
- status/stub
- domain/pretraining
title: Enhancing Model Safety through Pretraining Data Filtering
type: note
updated: '2026-05-10'
year: 2025
---

# Enhancing Model Safety through Pretraining Data Filtering

Safety-oriented pretraining data filtering to remove CBRN-related and other misuse-enabling content.

- **Authors**: Anthropic (Alignment Science)
- **Year**: 2025 (August)
- **URL**: [alignment.anthropic.com/2025/pretraining-data-filtering](https://alignment.anthropic.com/2025/pretraining-data-filtering/)
- **Type**: Alignment research blog post (not arXiv)

## Core contribution

Experiments with classifier-based identification and removal of CBRN (chemical, biological, radiological, nuclear) weapons content from pretraining data. Tests 6 classification methods: finetuned Constitutional classifier, prompted Constitutional classifier, holdout loss / "canary model", FastText, named-entity string matching, and others. Reduces harmful-capabilities eval accuracy by 33% relative to random baseline while preserving standard benchmark performance (MMLU, code, prose). Positions safety-oriented filtering earlier in the pipeline rather than relying solely on post-training refusal mechanisms.

## Related

- Companion: "Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs" — introduces Selective Gradient Masking (SGTM) ([alignment.anthropic.com/2025/selective-gradient-masking](https://alignment.anthropic.com/2025/selective-gradient-masking/))
