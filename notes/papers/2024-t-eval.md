---
arxiv: '2312.14033'
authors:
- Zehui Chen
- Weihua Du
- Wenwei Zhang
- Kuikun Liu
- Jiangning Liu
- Miao Zheng
- Jingming Zhuo
- Songyang Zhang
- Dahua Lin
- Kai Chen
- Feng Zhao
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2312.14033
  raw: null
  source: https://arxiv.org/abs/2312.14033
owner: blaz
read: false
slug: t-eval
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/evals
- domain/agents
title: 'T-Eval: Evaluating the Tool Utilization Capability of Large Language Models
  Step by Step'
type: note
updated: '2026-05-10'
year: 2024
---

# T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step

## Citation

- URL: https://arxiv.org/abs/2312.14033
- Authors: Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, Feng Zhao
- Year / venue: 2024 arXiv preprint
- arXiv: 2312.14033

## Short Summary

First intrinsic evaluation framework for tool-using LLMs. Decomposes tool-use into sub-tasks — selection, usage, planning — and evaluates each independently. Does for *model evaluation* what [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms|Quality Matters]] does for *data evaluation*: break tool-use into structural components rather than measuring end-to-end pass rate.

## Relevance To Poolside

Structural decomposition of tool-use evaluation. Could inform how we evaluate agent capabilities beyond aggregate pass rates.

## Related Notes

- [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]] — analogous decomposition for data quality
- [[concepts/task-specific-quality-decomposition]] — same structural pattern applied to data
- [[evals/bfcl-v3]], [[evals/bfcl-v4]] — related function-calling evals

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
