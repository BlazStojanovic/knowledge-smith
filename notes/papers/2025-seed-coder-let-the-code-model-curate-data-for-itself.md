---
arxiv: '2506.03524'
authors:
- ByteDance Seed
- Yuyu Zhang
- Jing Su
- Yifan Sun
- Chenguang Xi
- Xia Xiao
- Shen Zheng
- Anxiang Zhang
- Kaibo Liu
- Daoguang Zan
- Tao Sun
- Jinhua Zhu
- Shulin Xin
- Dong Huang
- Yetao Bai
- Lixin Dong
- Chao Li
- Jianchong Chen
- Hanzhi Zhou
- Yifan Huang
- Guanghan Ning
- Xierui Song
- Jiaze Chen
- Siyao Liu
- Kai Shen
- Liang Xiang
- Yonghui Wu
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.03524
  raw: '[[raw/papers/md/2025-seed-coder-let-the-code-model-curate-data-for-itself]]'
  source: https://arxiv.org/abs/2506.03524
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-seed-coder-let-the-code-model-curate-data-for-itself.md
raw_pdf: raw/papers/pdf/2025-seed-coder-let-the-code-model-curate-data-for-itself.pdf
read: false
slug: seed-coder-let-the-code-model-curate-data-for-itself
tags:
- type/paper
- status/stub
title: 'Seed-Coder: Let the Code Model Curate Data for Itself'
type: note
updated: '2026-05-11'
year: 2025
---

# Seed-Coder: Let the Code Model Curate Data for Itself

> *ByteDance Seed, Yuyu Zhang, Jing Su, Yifan Sun, Chenguang Xi, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Code data in large language model (LLM) pretraining is recognized crucial not only for code-related tasks but also for enhancing general intelligence of LLMs. Current open-source LLMs often heavily rely on human effort to produce their code pretraining data, such as employing hand-crafted filtering rules tailored to individual programming languages, or using human-annotated data to train quality filters. However, these approaches are inherently limited in scalability, prone to subjective biases, and costly to extend and maintain across diverse programming languages. To address these challenges, we introduce Seed-Coder, a series of open-source LLMs comprising base, instruct and reasoning models of 8B size, minimizing human involvement in data construction. Our code pretraining data is produced by a model-centric data pipeline, which predominantly leverages LLMs for scoring and filtering code data. The instruct model is further trained via supervised fine-tuning and preference optimization, and the reasoning model leverages Long-Chain-of-Thought (LongCoT) reinforcement learning to improve multi-step code reasoning. Seed-Coder achieves state-of-the-art results among open-source models of similar size and even surpasses some much larger models, demonstrating superior performance in code generation, code completion, code editing, code reasoning, and software engineering tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.03524>
- PDF: [[raw/papers/pdf/2025-seed-coder-let-the-code-model-curate-data-for-itself.pdf]]
- Raw markdown: [[raw/papers/md/2025-seed-coder-let-the-code-model-curate-data-for-itself]]
