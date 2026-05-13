---
arxiv: '2509.25531'
authors:
- Huu Nguyen
- Victor May
- Harsh Raj
- Marianna Nezhurina
- Yishan Wang
- Yanqi Luo
- Minh Chien Vu
- Taishi Nakamura
- Ken Tsui
- Van Khue Nguyen
- David Salinas
- Aleksandra Krasnodębska
- Christoph Schuhmann
- Mats Leon Richter
- Xuan-Son
- Vu
- Jenia Jitsev
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2509.25531
  raw: '[[raw/papers/md/2025-mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction]]'
  source: https://arxiv.org/abs/2509.25531
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction.md
raw_pdf: raw/papers/pdf/2025-mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction.pdf
read: false
slug: mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction
tags:
- type/paper
- status/stub
title: 'MixtureVitae: Open Web-Scale Pretraining Dataset With High Quality Instruction
  and Reasoning Data Built from Permissive-First Text Sources'
type: note
updated: '2026-05-11'
year: 2025
---

# MixtureVitae: Open Web-Scale Pretraining Dataset With High Quality Instruction and Reasoning Data Built from Permissive-First Text Sources

> *Huu Nguyen, Victor May, Harsh Raj, Marianna Nezhurina, Yishan Wang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

We present MixtureVitae, an open-access pretraining corpus built to minimize legal risk while providing strong downstream performance. MixtureVitae follows a permissive-first, risk-mitigated sourcing strategy that combines public-domain and permissively licensed text (e.g., CC-BY/Apache) with carefully justified low-risk additions (e.g., government works and EU TDM-eligible sources). MixtureVitae adopts a simple, single-stage pretraining recipe that integrates a large proportion of permissive synthetic instruction and reasoning data-signals typically introduced during post-training and generally scarce in permissive web corpora. We categorize all sources into a three-tier scheme that reflects varying risk levels and provide shard-level provenance metadata to enable risk-aware usage. In controlled experiments using the open-sci-ref training protocol (fixed architectures and hyperparameters; 50B and 300B token budgets across 130M-1.7B parameters), models trained on MixtureVitae consistently outperform other permissive datasets across a suite of standard benchmarks, and at the 1.7B-parameters/300B-tokens setting, they surpass FineWeb-Edu and approach DCLM late in training. Performance is particularly strong on MMLU and on math and code benchmarks: a 1.7B model pretrained on 300B MixtureVitae tokens matches or exceeds a strong 1.7B instruction-tuned baseline on GSM8K, HumanEval, and MBPP, despite using over 36 times fewer tokens (300B vs. ~11T). Supported by a thorough decontamination analysis, these results show that permissive-first data with high instruction and reasoning density, tiered by licensing and provenance-related risk, can provide a practical and risk-mitigated foundation for training capable LLMs, reducing reliance on broad web scrapes without sacrificing competitiveness. Code: https://github.com/ontocord/mixturevitae

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2509.25531>
- PDF: [[raw/papers/pdf/2025-mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction.pdf]]
- Raw markdown: [[raw/papers/md/2025-mixturevitae-open-web-scale-pretraining-dataset-with-high-quality-instruction]]
