---
arxiv: '2603.16177'
authors:
- Christina Baek
- Ricardo Pio Monti
- David Schwab
- Amro Abbas
- Rishabh Adiga
- Cody Blakeney
- Maximilian Böther
- Paul Burstein
- Aldo Gael Carranza
- Alvin Deng
- Parth Doshi
- Vineeth Dorna
- Alex Fang
- Tony Jiang
- Siddharth Joshi
- Brett W. Larsen
- Jason Chan Lee
- Katherine L. Mentzer
- Luke Merrick
- Haakon Mongstad
- Fan Pan
- Anshuman Suri
- Darren Teh
- Jason Telanoff
- Jack Urbanek
- Zhengping Wang
- Josh Wills
- Haoli Yin
- Aditi Raghunathan
- J. Zico Kolter
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
- Pratyush Maini
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.16177
  raw: '[[raw/papers/md/2026-finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data]]'
  source: https://arxiv.org/abs/2603.16177
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data.md
raw_pdf: raw/papers/pdf/2026-finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data.pdf
read: false
slug: finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data
tags:
- type/paper
- status/stub
title: 'The Finetuner''s Fallacy: When to Pretrain with Your Finetuning Data'
type: note
updated: '2026-05-11'
year: 2026
---

# The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data

> *Christina Baek, Ricardo Pio Monti, David Schwab, Amro Abbas, Rishabh Adiga, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Real-world model deployments demand strong performance on narrow domains where data is often scarce. Typically, practitioners finetune models to specialize them, but this risks overfitting to the domain and forgetting general knowledge. We study a simple strategy, specialized pretraining (SPT), where a small domain dataset, typically reserved for finetuning, is repeated starting from pretraining as a fraction of the total tokens. Across three specialized domains (ChemPile, MusicPile, and ProofPile), SPT improves domain performance and preserves general capabilities after finetuning compared to standard pretraining. In our experiments, SPT reduces the pretraining tokens needed to reach a given domain performance by up to 1.75x. These gains grow when the target domain is underrepresented in the pretraining corpus: on domains far from web text, a 1B SPT model outperforms a 3B standard pretrained model. Beyond these empirical gains, we derive overfitting scaling laws to guide practitioners in selecting the optimal domain-data repetition for a given pretraining compute budget. Our observations reveal the finetuner's fallacy: while finetuning may appear to be the cheapest path to domain adaptation, introducing specialized domain data during pretraining stretches its utility. SPT yields better specialized domain performance (via reduced overfitting across repeated exposures) and better general domain performance (via reduced forgetting during finetuning), ultimately achieving stronger results with fewer parameters and less total compute when amortized over inference. To get the most out of domain data, incorporate it as early in training as possible.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.16177>
- PDF: [[raw/papers/pdf/2026-finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data.pdf]]
- Raw markdown: [[raw/papers/md/2026-finetuner-s-fallacy-when-to-pretrain-with-your-finetuning-data]]
