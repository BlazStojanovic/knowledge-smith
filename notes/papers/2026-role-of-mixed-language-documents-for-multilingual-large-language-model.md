---
arxiv: '2601.00364'
authors:
- Jiandong Shao
- Raphael Tang
- Crystina Zhang
- Karin Sevegnani
- Pontus Stenetorp
- Jianfei Yang
- Yao Lu
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.00364
  raw: '[[raw/papers/md/2026-role-of-mixed-language-documents-for-multilingual-large-language-model]]'
  source: https://arxiv.org/abs/2601.00364
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-role-of-mixed-language-documents-for-multilingual-large-language-model.md
raw_pdf: raw/papers/pdf/2026-role-of-mixed-language-documents-for-multilingual-large-language-model.pdf
read: false
slug: role-of-mixed-language-documents-for-multilingual-large-language-model
tags:
- type/paper
- status/stub
- multilingual
- pretraining
- llm
title: The Role of Mixed-Language Documents for Multilingual Large Language Model
  Pretraining
type: note
updated: '2026-05-15'
year: 2026
---

# The Role of Mixed-Language Documents for Multilingual Large Language Model Pretraining

> *Jiandong Shao, Raphael Tang, Crystina Zhang, Karin Sevegnani, Pontus Stenetorp, et al.* — arXiv 2026

## TL;DR

Mixed-language documents are small but disproportionately important for translation. Removing bilingual data, only ~2% of the corpus, causes a 56% BLEU drop, while cross-lingual QA and reasoning stay mostly stable. The key signal is not code-switching → it is parallel text. Reintroducing parallel data restores 91% of translation performance, while code-switching helps little. **Key**: Translation needs token-level alignment; general reasoning may not. **Caveat**: No evals on maths/coding, only knowledge/reasoning. (tl;dr by Blaz.)

## Abstract

Multilingual large language models achieve impressive cross-lingual performance despite largely monolingual pretraining. While bilingual data in pretraining corpora is widely believed to enable these abilities, details of its contributions remain unclear. We investigate this question by pretraining models from scratch under controlled conditions, comparing the standard web corpus with a monolingual-only version that removes all multilingual documents. Despite constituting only 2% of the corpus, removing bilingual data causes translation performance to drop 56% in BLEU, while behaviour on cross-lingual QA and general reasoning tasks remains stable, with training curves largely overlapping the baseline. To understand this asymmetry, we categorize bilingual data into parallel (14%), code-switching (72%), and miscellaneous documents (14%) based on the semantic relevance of content in different languages. We then conduct granular ablations by reintroducing parallel or code-switching data into the monolingual-only corpus. Our experiments reveal that parallel data almost fully restores translation performance (91% of the unfiltered baseline), whereas code-switching contributes minimally. Other cross-lingual tasks remain largely unaffected by either type. These findings reveal that translation critically depends on systematic token-level alignments from parallel data, whereas cross-lingual understanding and reasoning appear to be achievable even without bilingual data.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.00364>
- PDF: [[raw/papers/pdf/2026-role-of-mixed-language-documents-for-multilingual-large-language-model.pdf]]
- Raw markdown: [[raw/papers/md/2026-role-of-mixed-language-documents-for-multilingual-large-language-model]]
