---
arxiv: '2605.20613'
authors:
- Guan Wang
- Changling Liu
- Chenyu Wang
- Cai Zhou
- Yuhao Sun
- Yifei Wu
- Shuai Zhen
- Luca Scimeca
- Yasin Abbasi Yadkori
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.20613
  raw: '[[raw/papers/md/2026-hrm-text-efficient-pretraining-beyond-scaling]]'
  source: https://arxiv.org/abs/2605.20613
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-hrm-text-efficient-pretraining-beyond-scaling.md
raw_pdf: raw/papers/pdf/2026-hrm-text-efficient-pretraining-beyond-scaling.pdf
read: false
slug: hrm-text-efficient-pretraining-beyond-scaling
tags:
- type/paper
- status/stub
title: 'HRM-Text: Efficient Pretraining Beyond Scaling'
type: note
updated: '2026-05-28'
year: 2026
---

# HRM-Text: Efficient Pretraining Beyond Scaling

> *Guan Wang, Changling Liu, Chenyu Wang, Cai Zhou, Yuhao Sun, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

The current pretraining paradigm for large language models relies on massive compute and internet-scale raw text, creating a significant barrier to foundational research. In contrast, biological systems demonstrate highly sample-efficient learning through multi-timescale processing, such as the functional organization of the frontoparietal loop. Taking this as inspiration, we introduce HRM-Text, which replaces standard Transformers with a Hierarchical Recurrent Model (HRM) that decouples computation into slow-evolving strategic and fast-evolving execution layers. To stabilize this deep recurrence for language modeling, we introduce MagicNorm and warmup deep credit assignment. Furthermore, instead of standard raw-text pretraining, we train exclusively on instruction-response pairs using a task-completion objective and PrefixLM masking. Serving as an empirical existence proof of efficient pretraining, a 1B-parameter HRM-Text model trained from scratch on only 40 billion unique tokens and $1,500 budget achieves 60.7% on MMLU, 81.9% on ARC-C, 82.2% on DROP, 84.5% on GSM8K, and 56.2% on MATH. Despite utilizing roughly 100-900x fewer training tokens and 96-432x less estimated compute than standard baselines, HRM-Text performs competitively with 2-7B parameter open models. These results demonstrate that co-designing architectures and objectives can radically reduce the compute-to-performance ratio, making pretraining from scratch accessible to the broader research community.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.20613>
- PDF: [[raw/papers/pdf/2026-hrm-text-efficient-pretraining-beyond-scaling.pdf]]
- Raw markdown: [[raw/papers/md/2026-hrm-text-efficient-pretraining-beyond-scaling]]
