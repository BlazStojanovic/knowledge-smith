---
arxiv: '2604.24432'
authors:
- Chenglong Chu
- Guorui Zhou
- Guowang Zhang
- Han Li
- Hao Peng
- Hongtao Cheng
- Jian Liang
- Jiangxia Cao
- Kun Gai
- Lingzhi Zhou
- Lu Ren
- Qi Zhang
- Ruiming Tang
- Ruitao Wang
- Xinchen Luo
- Yi Su
- Zhiyuan Liang
- Ziqi Wang
- Boyang Ding
- Chengru Song
- Dunju Zang
- Hui Wang
- Jiao Ou
- Jiaxin Deng
- Jijun Shi
- Jinghao Zhang
- Junmin Chen
- Lejian Ren
- Minxuan Lv
- Qianqian Wang
- Qigen Hu
- Shiyao Wang
- Siyang Mao
- Tao Wang
- Xingmei Wang
- Zhixin Ling
- Ziming Li
- Zixing Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.24432
  raw: '[[raw/papers/md/2026-kwai-summary-attention-technical-report]]'
  source: https://arxiv.org/abs/2604.24432
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-kwai-summary-attention-technical-report.md
raw_pdf: raw/papers/pdf/2026-kwai-summary-attention-technical-report.pdf
read: false
slug: kwai-summary-attention-technical-report
tags:
- type/paper
- status/stub
title: Kwai Summary Attention Technical Report
type: note
updated: '2026-05-11'
year: 2026
---

# Kwai Summary Attention Technical Report

> *Chenglong Chu, Guorui Zhou, Guowang Zhang, Han Li, Hao Peng, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Long-context ability, has become one of the most important iteration direction of next-generation Large Language Models, particularly in semantic understanding/reasoning, code agentic intelligence and recommendation system. However, the standard softmax attention exhibits quadratic time complexity with respect to sequence length. As the sequence length increases, this incurs substantial overhead in long-context settings, leading the training and inference costs of extremely long sequences deteriorate rapidly. Existing solutions mitigate this issue through two technique routings: i) Reducing the KV cache per layer, such as from the head-level compression GQA, and the embedding dimension-level compression MLA, but the KV cache remains linearly dependent on the sequence length at a 1:1 ratio. ii) Interleaving with KV Cache friendly architecture, such as local attention SWA, linear kernel GDN, but often involve trade-offs among KV Cache and long-context modeling effectiveness. Besides the two technique routings, we argue that there exists an intermediate path not well explored: {Maintaining a linear relationship between the KV cache and sequence length, but performing semantic-level compression through a specific ratio $k$}. This $O(n/k)$ path does not pursue a ``minimum KV cache'', but rather trades acceptable memory costs for complete, referential, and interpretable retention of long distant dependency. Motivated by this, we propose Kwai Summary Attention (KSA), a novel attention mechanism that reduces sequence modeling cost by compressing historical contexts into learnable summary tokens.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.24432>
- PDF: [[raw/papers/pdf/2026-kwai-summary-attention-technical-report.pdf]]
- Raw markdown: [[raw/papers/md/2026-kwai-summary-attention-technical-report]]
