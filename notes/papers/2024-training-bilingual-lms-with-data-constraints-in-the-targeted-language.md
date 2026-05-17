---
arxiv: '2411.12986'
authors:
- Skyler Seto
- Maartje ter Hoeve
- Richard He Bai
- Natalie Schluter
- David Grangier
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2411.12986
  raw: '[[raw/papers/md/2024-training-bilingual-lms-with-data-constraints-in-the-targeted-language]]'
  source: https://arxiv.org/abs/2411.12986
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-training-bilingual-lms-with-data-constraints-in-the-targeted-language.md
raw_pdf: raw/papers/pdf/2024-training-bilingual-lms-with-data-constraints-in-the-targeted-language.pdf
read: false
slug: training-bilingual-lms-with-data-constraints-in-the-targeted-language
tags:
- type/paper
- status/stub
- multilingual
- pretraining
- llm
title: Training Bilingual LMs with Data Constraints in the Targeted Language
type: note
updated: '2026-05-15'
venue: Findings of ACL 2025
year: 2024
---

# Training Bilingual LMs with Data Constraints in the Targeted Language

> *Skyler Seto, Maartje ter Hoeve, Richard He Bai, Natalie Schluter, David Grangier* — Findings of ACL 2025 (arXiv 2024)

## TL;DR

When target-language data is scarce, a strong auxiliary language can help, especially when the languages are close. Better English or other high-quality auxiliary data can improve the target language without changing the model or training objective. But model scaling alone does not solve target-language scarcity; data quality, language relatedness, and sampling strategy matter. **Caveat**: They want target-language (e.g., German) performance, English is the crutch/auxiliary. Also no evals on maths/coding, only knowledge/reasoning. (tl;dr by Blaz.)

## Abstract

Large language models are trained on massive scrapes of the web, as required by current scaling laws. Most progress is made for English, given its abundance of high-quality pretraining data. For most other languages, however, such high quality pretraining data is unavailable. In this work, we study how to boost pretrained model performance in a target language with insufficient pretraining data for training a high performing language model, by enlisting data from an auxiliary language for which high quality data is available. We study this by quantifying the performance gap between training with data in a data-rich auxiliary language compared with training in the target language, exploring the benefits of translation systems, studying the limitations of model scaling when data is limited in the target languages, and proposing new methods for upsampling data from the auxiliary language. Our results show that stronger auxiliary datasets result in performance gains without modification to the model or training objective for close languages, and, in particular, that performance gains due to the development of more information-rich English pretraining datasets can extend to targeted language settings with limited data.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2411.12986>
- PDF: [[raw/papers/pdf/2024-training-bilingual-lms-with-data-constraints-in-the-targeted-language.pdf]]
- Raw markdown: [[raw/papers/md/2024-training-bilingual-lms-with-data-constraints-in-the-targeted-language]]
