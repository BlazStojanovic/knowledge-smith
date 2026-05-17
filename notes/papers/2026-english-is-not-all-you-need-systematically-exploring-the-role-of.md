---
arxiv: '2604.13286'
authors:
- Mehak Dhaliwal
- Shashwat Chaurasia
- Yao Qin
- Dezhi Hong
- Thomas Butler
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.13286
  raw: '[[raw/papers/md/2026-english-is-not-all-you-need-systematically-exploring-the-role-of]]'
  source: https://arxiv.org/abs/2604.13286
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-english-is-not-all-you-need-systematically-exploring-the-role-of.md
raw_pdf: raw/papers/pdf/2026-english-is-not-all-you-need-systematically-exploring-the-role-of.pdf
read: false
slug: english-is-not-all-you-need-systematically-exploring-the-role-of
tags:
- type/paper
- status/stub
- multilingual
- fine-tuning
- llm
title: 'English is Not All You Need: Systematically Exploring the Role of Multilinguality
  in LLM Post-Training'
type: note
updated: '2026-05-15'
year: 2026
---

# English is Not All You Need: Systematically Exploring the Role of Multilinguality in LLM Post-Training

> *Mehak Dhaliwal, Shashwat Chaurasia, Yao Qin, Dezhi Hong, Thomas Butler* — arXiv 2026

## TL;DR

English-only post-training is usually a weak default for globally deployed LLMs. In 220 fine-tuning runs across math reasoning and API calling, adding multilingual data generally improves or preserves performance, with low-resource languages benefiting most. Even adding one non-English language to English-only SFT lifted English by a median 3.4% on math reasoning and 0.88% on API calling, winning in 75% of configs. **Caveat**: very small models can hit capacity limits, especially on structured API-calling tasks. (tl;dr by Blaz.)

## Abstract

Despite the widespread multilingual deployment of large language models, post-training pipelines remain predominantly English-centric, contributing to performance disparities across languages. We present a systematic, controlled study of the interplay between training language coverage, model scale, and task domain, based on 220 supervised fine-tuning runs on parallel translated multilingual data mixtures spanning mathematical reasoning and API calling tasks, with models up to 8B parameters. We find that increasing language coverage during post-training is largely beneficial across tasks and model scales, with low-resource languages benefiting the most and high-resource languages plateauing rather than degrading. Even minimal multilinguality helps: incorporating a single non-English language improves both English performance and cross-lingual generalization, making English-only post-training largely suboptimal. Moreover, at sufficient language diversity, zero-shot cross-lingual transfer can match or exceed the effects of direct language inclusion in a low-diversity setting, although gains remain limited for typologically distant, low-resource languages.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.13286>
- PDF: [[raw/papers/pdf/2026-english-is-not-all-you-need-systematically-exploring-the-role-of.pdf]]
- Raw markdown: [[raw/papers/md/2026-english-is-not-all-you-need-systematically-exploring-the-role-of]]
