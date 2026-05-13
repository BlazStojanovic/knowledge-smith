---
arxiv: '2212.10559'
authors:
- Damai Dai
- Yutao Sun
- Li Dong
- Yaru Hao
- Shuming Ma
- Zhifang Sui
- Furu Wei
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2212.10559
  raw: null
  source: https://arxiv.org/abs/2212.10559
owner: blaz
read: false
slug: icl-as-gradient-descent
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/reasoning
title: Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent
  as Meta-Optimizers
type: note
updated: '2026-05-10'
year: 2023
---

# Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers

## Citation

- URL: https://arxiv.org/abs/2212.10559
- Authors: Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, Furu Wei
- Year / venue: ACL 2023 Findings
- arXiv: 2212.10559

## Short Summary

Shows that Transformer attention in ICL is mathematically dual to gradient descent: in-context examples produce implicit weight updates similar to fine-tuning. Implies that properties making a good few-shot example overlap with properties making good training data. See also Von Oswald et al. 2023 (ICML, arXiv 2212.07677) for a more formal treatment.

## Relevance To Poolside

Theoretical foundation for ICE in [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]]. Broader implication: bridges the difficulty/utility literature (IFD, LESS, Dataset Cartography) with in-context evaluation — the same data properties may matter for both.

## Related Notes

- [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]] — ICE method motivated by this connection
- [[concepts/difficulty]] — behavioural difficulty signals share structure with ICL effectiveness

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
