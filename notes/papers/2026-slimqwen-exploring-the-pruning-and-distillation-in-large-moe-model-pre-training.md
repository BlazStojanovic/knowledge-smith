---
arxiv: '2605.08738'
authors:
- Shengkun Tang
- Zekun Wang
- Bo Zheng
- Liangyu Wang
- Rui Men
- Siqi Zhang
- Xiulong Yuan
- Zihan Qiu
- Zhiqiang Shen
- Dayiheng Liu
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.08738
  raw: '[[raw/papers/md/2026-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training]]'
  source: https://arxiv.org/abs/2605.08738
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training.md
raw_pdf: raw/papers/pdf/2026-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training.pdf
read: false
slug: slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training
tags:
- type/paper
- status/stub
- mixture-of-experts
- distillation
- pretraining
- llm
title: 'SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training'
type: note
updated: '2026-05-15'
year: 2026
---

# SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training

> *Shengkun Tang, Zekun Wang, Bo Zheng, Liangyu Wang, Rui Men, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Structured pruning and knowledge distillation (KD) are typical techniques for compressing large language models, but it remains unclear how they should be applied at pretraining scale, especially to recent mixture-of-experts (MoE) models. In this work, we systematically study MoE compression in large-scale pretraining, focusing on three key questions: whether pruning provides a better initialization than training from scratch, how expert compression choices affect the final model after continued training, and which training strategy is most effective. We have the following findings: First, across depth, width, and expert compression, pruning a pretrained MoE consistently outperforms training the target architecture from scratch under the same training budget. Second, different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining. Motivated by this, we introduce a simple partial-preservation expert merging strategy that improves downstream performance across most benchmarks. Third, combining KD with the language modeling loss outperforms KD alone, particularly on knowledge-intensive tasks. We further propose multi-token prediction (MTP) distillation, which yields consistent gains. Finally, given the same training tokens, progressive pruning schedules outperform one-shot compression, suggesting that gradual architecture transitions lead to better optimization trajectories. Putting it all together, we compress Qwen3-Next-80A3B to a 23A2B model that retains competitive performance. These results offer practical guidance for efficient MoE compression at scale.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.08738>
- PDF: [[raw/papers/pdf/2026-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training.pdf]]
- Raw markdown: [[raw/papers/md/2026-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre-training]]
