---
arxiv: '2509.19249'
authors:
- Siheng Li
- Kejiao Li
- Zenan Xu
- Guanhua Huang
- Evander Yang
- Kun Li
- Haoyuan Wu
- Jiajia Wu
- Zihao Zheng
- Chenchen Zhang
- Kun Shi
- Kyrierl Deng
- Qi Yi
- Ruibin Xiong
- Tingqiang Xu
- Yuhao Jiang
- Jianfeng Yan
- Yuyuan Zeng
- Guanghui Xu
- Jinbao Xue
- Zhijiang Xu
- Zheng Fang
- Shuai Li
- Qibin Liu
- Xiaoxue Li
- Zhuoyu Li
- Yangyu Tao
- Fei Gao
- Cheng Jiang
- Bo Chao Wang
- Kai Liu
- Jianchen Zhu
- Wai Lam
- Wayyt Wang
- Bo Zhou
- Di Wang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2509.19249
  raw: '[[raw/papers/md/2025-reinforcement-learning-on-pre-training-data]]'
  source: https://arxiv.org/abs/2509.19249
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-reinforcement-learning-on-pre-training-data.md
raw_pdf: raw/papers/pdf/2025-reinforcement-learning-on-pre-training-data.pdf
read: false
slug: reinforcement-learning-on-pre-training-data
tags:
- type/paper
- status/stub
title: Reinforcement Learning on Pre-Training Data
type: note
updated: '2026-05-11'
year: 2025
---

# Reinforcement Learning on Pre-Training Data

> *Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2509.19249>
- PDF: [[raw/papers/pdf/2025-reinforcement-learning-on-pre-training-data.pdf]]
- Raw markdown: [[raw/papers/md/2025-reinforcement-learning-on-pre-training-data]]
