---
arxiv: '2604.11035'
authors:
- Yifan Yu
- Yuqing Jian
- Junxiong Wang
- Zhongzhu Zhou
- Donglin Zhuang
- Xinyu Fang
- Sri Yanamandra
- Xiaoxia Wu
- Qingyang Wu
- Shuaiwen Leon Song
- Tri Dao
- Ben Athiwaratkun
- James Zou
- Fan Lai
- Chenfeng Xu
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.11035
  raw: '[[raw/papers/md/2026-introspective-diffusion-language-models]]'
  source: https://arxiv.org/abs/2604.11035
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-introspective-diffusion-language-models.md
raw_pdf: raw/papers/pdf/2026-introspective-diffusion-language-models.pdf
read: false
slug: introspective-diffusion-language-models
tags:
- type/paper
- status/stub
title: Introspective Diffusion Language Models
type: note
updated: '2026-05-11'
year: 2026
---

# Introspective Diffusion Language Models

> *Yifan Yu, Yuqing Jian, Junxiong Wang, Zhongzhu Zhou, Donglin Zhuang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Diffusion language models promise parallel generation, yet still lag behind autoregressive (AR) models in quality. We stem this gap to a failure of introspective consistency: AR models agree with their own generations, while DLMs often do not. We define the introspective acceptance rate, which measures whether a model accepts its previously generated tokens. This reveals why AR training has a structural advantage: causal masking and logit shifting implicitly enforce introspective consistency. Motivated by this observation, we introduce Introspective Diffusion Language Model (I-DLM), a paradigm that retains diffusion-style parallel decoding while inheriting the introspective consistency of AR training. I-DLM uses a novel introspective strided decoding (ISD) algorithm, which enables the model to verify previously generated tokens while advancing new ones in the same forward pass. From a systems standpoint, we build I-DLM inference engine on AR-inherited optimizations and further customize it with a stationary-batch scheduler. To the best of our knowledge, I-DLM is the first DLM to match the quality of its same-scale AR counterpart while outperforming prior DLMs in both model quality and practical serving efficiency across 15 benchmarks. It reaches 69.6 on AIME-24 and 45.7 on LiveCodeBench-v6, exceeding LLaDA-2.1-mini (16B) by more than 26 and 15 points, respectively. Beyond quality, I-DLM is designed for the growing demand of large-concurrency serving, delivering about 3x higher throughput than prior state-of-the-art DLMs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.11035>
- PDF: [[raw/papers/pdf/2026-introspective-diffusion-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2026-introspective-diffusion-language-models]]
