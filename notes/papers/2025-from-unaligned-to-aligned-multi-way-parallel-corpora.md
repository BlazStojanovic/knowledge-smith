---
authors:
- Yingli Shen
- Wen Lai
- Shuo Wang
- Ge Gao
- Kangyang Luo
- Alexander Fraser
- Maosong Sun
created: '2026-05-15'
kind: paper
links:
  code: https://github.com/yl-shen/multi-way-llm
  paper: null
  raw: '[[raw/papers/md/2025-from-unaligned-to-aligned-multi-way-parallel-corpora]]'
  source: https://aclanthology.org/2025.emnlp-main.374/
owner: blaz
parser: read
raw_md: raw/papers/md/2025-from-unaligned-to-aligned-multi-way-parallel-corpora.md
raw_pdf: raw/papers/pdf/2025-from-unaligned-to-aligned-multi-way-parallel-corpora.pdf
read: false
slug: from-unaligned-to-aligned-multi-way-parallel-corpora
tags:
- type/paper
- status/stub
- multilingual
- pretraining
- llm
title: 'From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel
  Corpora'
type: note
updated: '2026-05-15'
venue: EMNLP 2025
year: 2025
---

# From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora

> *Yingli Shen, Wen Lai, Shuo Wang et al. (Tsinghua, TU Munich, Minzu University)* — EMNLP 2025

## TL;DR

Multi-way aligned corpora beat unaligned multilingual text. The paper introduces TED2025 [not publicly available], a human-translated corpus spanning 113 languages with up to 50 aligned languages per example, and shows that continued pretraining/instruction tuning on this aligned data outperforms unaligned multilingual data across six benchmarks. Alignment is a scaling lever for multilinguality, especially low-resource transfer. **Caveat**: No evals on maths/coding, only knowledge/reasoning. (tl;dr by Blaz.)

## Abstract

Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multi-way parallel data consistently outperform those trained on unaligned multilingual data.

## Notes

(your synthesis — anything beyond the abstract belongs here)

## Source

- PDF: [[raw/papers/pdf/2025-from-unaligned-to-aligned-multi-way-parallel-corpora.pdf]]
- Source: <https://aclanthology.org/2025.emnlp-main.374/>
- Code: <https://github.com/yl-shen/multi-way-llm>
