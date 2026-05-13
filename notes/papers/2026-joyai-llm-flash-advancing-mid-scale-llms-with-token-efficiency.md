---
arxiv: '2604.03044'
authors:
- Aichen Cai
- Anmeng Zhang
- Anyu Li
- Bo Zhang
- Bohua Cai
- Chang Li
- Changjian Jiang
- Changkai Lu
- Chao Xue
- Chaocai Liang
- Cheng Zhang
- Dongkai Liu
- Fei Wang
- Guoqiang Huang
- Haijian Ke
- Han Lin
- Hao Wang
- Ji Miao
- Jiacheng Zhang
- Jialong Shi
- Jifeng Zhu
- Jingjing Qian
- Junhui Luo
- Junwu Xiong
- Lam So
- Liang Huang
- Ming Ke
- Mingyang Li
- Panfeng Shi
- Peng Hao
- Qi Wang
- Qian Lai
- Qiaoqiao Yuan
- Qingyu Yin
- Qiong Cao
- Qixiang Wang
- Rongcheng Bian
- Rongduo Han
- Shaoqiang Zheng
- Shi Hu
- Shi Suo
- Shijie Ren
- Shijin Zhang
- Shiying Fan
- Shuai Xie
- Tianyi Zhang
- Wei Liu
- Wentao Tan
- Xianghan Meng
- Xiaodong He
- Xing Pan
- Xiran Wang
- Xuyang Peng
- Ya Zhang
- Yang Liu
- Yangyang Duan
- Yanxu Chen
- Yicheng Gong
- Yidan Huang
- Yifei Liu
- Yinhao Bai
- Yongqiang Liu
- Yuesong Zhang
- Yuqi Zhang
- Zerui Xie
- Zhenfang Wang
- Zhennan Shen
- Zheyuan Liu
- Zhuwei Zeng
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.03044
  raw: '[[raw/papers/md/2026-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency]]'
  source: https://arxiv.org/abs/2604.03044
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency.md
raw_pdf: raw/papers/pdf/2026-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency.pdf
read: false
slug: joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency
tags:
- type/paper
- status/stub
title: 'JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency'
type: note
updated: '2026-05-11'
year: 2026
---

# JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency

> *Aichen Cai, Anmeng Zhang, Anyu Li, Bo Zhang, Bohua Cai, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We introduce JoyAI-LLM Flash, an efficient Mixture-of-Experts (MoE) language model designed to redefine the trade-off between strong performance and token efficiency in the sub-50B parameter regime. JoyAI-LLM Flash is pretrained on a massive corpus of 20 trillion tokens and further optimized through a rigorous post-training pipeline, including supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and large-scale reinforcement learning (RL) across diverse environments. To improve token efficiency, JoyAI-LLM Flash strategically balances \emph{thinking} and \emph{non-thinking} cognitive modes and introduces FiberPO, a novel RL algorithm inspired by fibration theory that decomposes trust-region maintenance into global and local components, providing unified multi-scale stability control for LLM policy optimization. To enhance architectural sparsity, the model comprises 48B total parameters while activating only 2.7B parameters per forward pass, achieving a substantially higher sparsity ratio than contemporary industry leading models of comparable scale. To further improve inference throughput, we adopt a joint training-inference co-design that incorporates dense Multi-Token Prediction (MTP) and Quantization-Aware Training (QAT). We release the checkpoints for both JoyAI-LLM-48B-A3B Base and its post-trained variants on Hugging Face to support the open-source community.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.03044>
- PDF: [[raw/papers/pdf/2026-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency.pdf]]
- Raw markdown: [[raw/papers/md/2026-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency]]
