---
arxiv: '2306.05301'
authors:
- Qiaoyu Tang
- Ziliang Deng
- Hongyu Lin
- Xianpei Han
- Qiao Liang
- Boxi Cao
- Le Sun
created: 2026-04-23
kind: paper
links:
  code: https://github.com/tangqiaoyu/ToolAlpaca
  paper: https://arxiv.org/abs/2306.05301
  raw: https://arxiv.org/pdf/2306.05301
  source: https://arxiv.org/abs/2306.05301
owner: blaz
read: false
slug: toolalpaca
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/synth-data
- domain/agents
title: 'ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated
  Cases'
type: note
updated: '2026-05-10'
year: 2023
---

# ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases

## Citation

- URL: https://arxiv.org/abs/2306.05301
- PDF: https://arxiv.org/pdf/2306.05301
- Authors: Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun
- Year / venue: 2023 arXiv preprint
- arXiv: 2306.05301

## Short Summary

Generates a compact tool-use dataset using 2.3K synthesized APIs with 4.2K training instances (1–2 API calls per instance). Demonstrates that even small synthetic tool-use datasets can train generalizable tool-use capabilities. Used as secondary benchmark in [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]], where 44.8% of instances were found to contain errors — cleaner than ToolBench but still substantially noisy.

## Relevance To Poolside

Our interpretation: keep this as an unread source for future grounding. Interesting as a contrast to ToolBench — synthesized APIs vs real APIs produce different quality profiles.

## Related Notes

- [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]] — quality audit finding 44.8% error rate
- [[notes/papers/2024-toolbench]] — contrasting real-API approach

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
