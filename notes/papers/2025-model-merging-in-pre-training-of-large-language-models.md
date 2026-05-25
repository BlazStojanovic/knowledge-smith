---
arxiv: '2505.12082'
authors:
- Yunshui Li
- Yiyuan Ma
- Shen Yan
- Chaoyi Zhang
- Jing Liu
- Jianqiao Lu
- Ziwen Xu
- Mengzhao Chen
- Minrui Wang
- Shiyi Zhan
- Jin Ma
- Xunhao Lai
- Deyi Liu
- Yao Luo
- Xingyan Bin
- Hongbin Ren
- Mingji Han
- Wenhao Hao
- Bairen Yi
- LingJun Liu
- Bole Ma
- Xiaoying Jia
- Xun Zhou
- Siyuan Qiao
- Liang Xiang
- Yonghui Wu
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.12082
  raw: '[[raw/papers/md/2025-model-merging-in-pre-training-of-large-language-models]]'
  source: https://arxiv.org/abs/2505.12082
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-model-merging-in-pre-training-of-large-language-models.md
raw_pdf: raw/papers/pdf/2025-model-merging-in-pre-training-of-large-language-models.pdf
read: false
slug: model-merging-in-pre-training-of-large-language-models
tags:
- type/paper
- status/stub
title: Model Merging in Pre-training of Large Language Models
type: note
updated: '2026-05-25'
year: 2025
---

# Model Merging in Pre-training of Large Language Models

> *Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored. In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process. Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters, we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior. These improvements lead to both more efficient model development and significantly lower training costs. Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications. Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging.

## Notes

Pointer from Seonghyeon Kim in `#project-laguna` 2026-05-18, replying to Marah Abdin's plateau analysis (see `experiments/plateau-analysis-laguna-xs1-stage1.md` in the synthesis vault): "would be interesting if we could get the results from the model annealed from specific steps." The relevant link: this paper claims that merging stable-stage (constant-LR) checkpoints lets you predict post-anneal / cooldown behaviour, which is exactly the bridge between Marah's stable-phase churn analysis (says model is done at B) and George Grigorev's TPP scaling-law analysis (says XSv1 30T post-cooldown still sits on the Pareto frontier). If true, we can interrogate "what would cooldown buy?" without paying for a full cooldown run on every stable-stage checkpoint.

To compare against:
- `experiments/plateau-analysis-laguna-xs1-stage1.md` — Marah + Jianxiao's stable-stage churn analysis.
- `experiments/max-usable-tpp-scaling-law.md` — George's TPP scaling-law fit.
- `concepts/cooldown-as-knowledge-extraction.md` — current cooldown framing.

## Source

- arXiv: <https://arxiv.org/abs/2505.12082>
- PDF: [[raw/papers/pdf/2025-model-merging-in-pre-training-of-large-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2025-model-merging-in-pre-training-of-large-language-models]]
