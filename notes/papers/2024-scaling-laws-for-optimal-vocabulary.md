---
arxiv: '2407.13623'
authors:
- Chaofan Tao
- Qian Liu
- Longxu Dou
- Niklas Muennighoff
- Zhongwei Wan
- Ping Luo
- Min Lin
- Ngai Wong
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.13623
  raw: '[[raw/papers/md/2024-scaling-laws-for-optimal-vocabulary]]'
  source: https://arxiv.org/abs/2407.13623
owner: blaz
raw_pdf: raw/papers/pdf/2024-scaling-laws-for-optimal-vocabulary.pdf
read: false
slug: scaling-laws-for-optimal-vocabulary
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/training
title: 'Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies'
type: note
updated: '2026-05-10'
year: 2024
---

# Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies

## Citation

- URL: https://arxiv.org/abs/2407.13623
- Authors: Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong
- Year / venue: 2024 / NeurIPS 2024
- arXiv: 2407.13623
- **Raw**: [[raw/papers/pdf/2024-scaling-laws-for-optimal-vocabulary.pdf]]

## Core Claim

Optimal vocabulary size scales with compute budget. Non-vocabulary parameters scale faster than vocabulary parameters (exponent ~0.84 vs implied slower for vocab). Under-sized vocabularies waste compute on redundant byte-level processing; over-sized vocabularies waste parameters on rare tokens.

## Key Ideas

- Three approaches to determine optimal V: IsoFLOP analysis, derivative-based estimation, parametric fit of L(N_nonvocab, V, D)
- Practical shift: GPT-2 used V=50K → LLaMA 3 uses V=128K. This increase is justified by both scaling law analysis and multilingual coverage
- The optimal V depends on C (total compute), not just corpus statistics. At larger compute budgets, larger vocabularies become worthwhile
- Vocabulary parameters are "cheaper" per parameter than non-vocab parameters for reducing loss, but their contribution saturates faster

## Relevance To Poolside

Directly applicable to tokenizer decisions for new model training. If Poolside scales to larger models, the vocabulary size should scale accordingly.

## Related Notes

- [[concepts/architecture-scaling]] — vocabulary as an architecture component
- [[maps/scaling-laws/landscape]] — architecture components domain
