---
arxiv: '2310.01377'
authors:
- Ganqu Cui
- Lifan Yuan
- Ning Ding
- Guanming Yao
- Wei Zhu
- Yuan Ni
- Guotong Xie
- Zhiyuan Liu
- Maosong Sun
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2310.01377
  raw: null
  source: https://arxiv.org/abs/2310.01377
owner: blaz
read: false
slug: ultrafeedback-boosting-language-models-with-scaled-ai-feedback
tags:
- type/paper
- source/primary
- status/stub
- domain/synth-data
- stage/dpo
title: 'UltraFeedback: Boosting Language Models with Scaled AI Feedback'
type: note
updated: '2026-05-10'
year: 2024
---

# UltraFeedback: Boosting Language Models with Scaled AI Feedback

Large-scale AI feedback dataset for preference alignment, evaluating outputs on helpfulness, honesty, and other rubrics.

- **Authors**: Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, Maosong Sun
- **Venue**: ICML 2024
- **arXiv**: [2310.01377](https://arxiv.org/abs/2310.01377)
- **Raw**: [[raw/papers/pdf/2024-ultrafeedback]]

## Core contribution

Constructs a large-scale, high-quality AI feedback dataset by collecting responses from multiple LLMs and having GPT-4 annotate preferences across multiple rubrics (helpfulness, honesty, instruction-following, truthfulness). Shifts the control mechanism from manual prompt engineering to scalable automated preference filtering.

## Connections

- Related: [[notes/papers/2024-codeultrafeedback]] (code-domain variant)
