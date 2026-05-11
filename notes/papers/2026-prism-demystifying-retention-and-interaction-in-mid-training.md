---
arxiv: '2603.17074'
authors:
- Bharat Runwal
- Ashish Agrawal
- Anurag Roy
- Rameswar Panda
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.17074
  raw: '[[raw/papers/md/2026-prism-demystifying-retention-and-interaction-in-mid-training]]'
  source: https://arxiv.org/abs/2603.17074
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-prism-demystifying-retention-and-interaction-in-mid-training.md
raw_pdf: raw/papers/pdf/2026-prism-demystifying-retention-and-interaction-in-mid-training.pdf
read: false
slug: prism-demystifying-retention-and-interaction-in-mid-training
tags:
- type/paper
- llm
- fine-tuning
- pretraining
- status/stub
title: 'PRISM: Demystifying Retention and Interaction in Mid-Training'
type: note
updated: '2026-05-09'
venue: null
year: 2026
---

# PRISM: Demystifying Retention and Interaction in Mid-Training

> *Bharat Runwal, Ashish Agrawal, Anurag Roy…* — arXiv 2603.17074, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We present PRISM, a comprehensive empirical study of mid-training design choices for large language models. Through controlled experiments across seven base models spanning four families (Granite, LLaMA, Mistral, Nemotron-H), two architecture types (dense Transformer and attention-Mamba hybrid), and scales from 3B to 24B parameters, we show that mid-training on approximately 27B high-quality tokens yields consistent gains of +15 to +40 points on math, +5 to +12 points on code, and +6 to +13 points on science benchmarks while preserving general performance. The full PRISM to RL pipeline improves macro-average across six reasoning benchmarks from under 12 to 29-42 (a 3-4x improvement), whereas RL applied directly to most of the base models remains substantially less effective, with AIME scores near zero. Data composition matters most at mid-training, not RL: including science data during mid-training unlocks +17 to +28 point GPQA-Diamond gains during RL, while changing the RL mix produces less than 2 point differences. Mechanistically, mid-training densely restructures over 90% of model weights, while RL makes sparse, front-loaded refinements to approximately 5% of parameters. Representation analysis (CKA) confirms that RL consistently preserves mid-training's representational geometry (over 0.998 CKA) across architectures. Crucially, RL applies identical weight changes regardless of starting point, yet only succeeds on mid-trained models, consistent with mid-training placing the model in a configuration from which RL can effectively improve performance. Our results demonstrate that retention-aware mid-training is highly effective for reliable reasoning enhancement and provide practical guidance for designing robust mid-training pipelines.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-prism-demystifying-retention-and-interaction-in-mid-training]]
- PDF: [[raw/papers/pdf/2026-prism-demystifying-retention-and-interaction-in-mid-training.pdf]]
- arXiv: <https://arxiv.org/abs/2603.17074>
