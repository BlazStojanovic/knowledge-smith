---
arxiv: '2601.17858'
authors:
- Jiapeng Wang
- Changxin Tian
- Kunlong Chen
- Ziqi Liu
- Jiaxin Mao
- Wayne Xin Zhao
- Zhiqiang Zhang
- Jun Zhou
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.17858
  raw: '[[raw/papers/md/2026-mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging]]'
  source: https://arxiv.org/abs/2601.17858
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging.md
raw_pdf: raw/papers/pdf/2026-mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging.pdf
read: false
slug: mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging
tags:
- type/paper
- status/stub
title: 'MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging'
type: note
updated: '2026-05-11'
year: 2026
---

# MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging

> *Jiapeng Wang, Changxin Tian, Kunlong Chen, Ziqi Liu, Jiaxin Mao, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Optimizing data mixtures is essential for unlocking the full potential of large language models (LLMs), yet identifying the optimal composition remains computationally prohibitive due to reliance on heuristic trials or expensive proxy training. To address this, we introduce \textbf{MergeMix}, a novel approach that efficiently determines optimal data mixing ratios by repurposing model merging weights as a high-fidelity, low-cost performance proxy. By training domain-specific experts on minimal tokens and optimizing their merging weights against downstream benchmarks, MergeMix effectively optimizes the performance of data mixtures without incurring the cost of full-scale training. Extensive experiments on models with 8B and 16B parameters validate that MergeMix achieves performance comparable to or surpassing exhaustive manual tuning while drastically reducing search costs. Furthermore, MergeMix exhibits high rank consistency (Spearman $ρ> 0.9$) and strong cross-scale transferability, offering a scalable, automated solution for data mixture optimization.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.17858>
- PDF: [[raw/papers/pdf/2026-mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging.pdf]]
- Raw markdown: [[raw/papers/md/2026-mergemix-optimizing-mid-training-data-mixtures-via-learnable-model-merging]]
