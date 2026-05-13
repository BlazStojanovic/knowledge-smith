---
arxiv: '2602.15210'
authors:
- DatologyAI
- ':'
- Aldo Gael Carranza
- Kaleigh Mentzer
- Ricardo Pio Monti
- Alex Fang
- Alvin Deng
- Amro Abbas
- Anshuman Suri
- Brett Larsen
- Cody Blakeney
- Darren Teh
- David Schwab
- Diego Kiner
- Fan Pan
- Haakon Mongstad
- Haoli Yin
- Jack Urbanek
- Jason Lee
- Jason Telanoff
- Josh Wills
- Luke Merrick
- Maximilian Böther
- Parth Doshi
- Paul Burstein
- Pratyush Maini
- Rishabh Adiga
- Siddharth Joshi
- Spandan Das
- Tony Jiang
- Vineeth Dorna
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.15210
  raw: '[[raw/papers/md/2026-berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset]]'
  source: https://arxiv.org/abs/2602.15210
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset.md
raw_pdf: raw/papers/pdf/2026-berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset.pdf
read: false
slug: berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset
tags:
- type/paper
- status/stub
title: 'ÜberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset'
type: note
updated: '2026-05-11'
year: 2026
---

# ÜberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset

> *DatologyAI, :, Aldo Gael Carranza, Kaleigh Mentzer, Ricardo Pio Monti, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Multilinguality is a core capability for modern foundation models, yet training high-quality multilingual models remains challenging due to uneven data availability across languages. A further challenge is the performance interference that can arise from joint multilingual training, commonly referred to as the "curse of multilinguality". We study multilingual data curation across thirteen languages and find that many reported regressions are not inherent to multilingual scaling but instead stem from correctable deficiencies in data quality and composition rather than fundamental capacity limits. In controlled bilingual experiments, improving data quality for any single language benefits others: curating English improves non-English performance in 12 of 13 languages, while curating non-English yields reciprocal improvements in English. Bespoke per-language curation produces substantially larger within-language improvements. Extending these findings to large-scale general-purpose training mixtures, we show that curated multilingual allocations comprising under 8% of total tokens remain remarkably effective. We operationalize this approach within an effort that produced a 20T-token pretraining corpus derived entirely from public sources. Models with 3B and 8B parameters trained on a 1T-token random subset achieve competitive multilingual accuracy with 4-10x fewer training FLOPs than strong public baselines, establishing a new Pareto frontier in multilingual performance versus compute. Moreover, these benefits extend to frontier model scale: the 20T-token corpus served as part of the pretraining dataset for Trinity Large (400B/A13B), which exhibits strong multilingual performance relative to its training FLOPs. These results show that targeted, per-language data curation mitigates multilingual interference and enables compute-efficient multilingual scaling.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.15210>
- PDF: [[raw/papers/pdf/2026-berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset.pdf]]
- Raw markdown: [[raw/papers/md/2026-berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset]]
