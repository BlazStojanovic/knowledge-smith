---
arxiv: '2605.12366'
authors:
- Sam Martin
- Fabien Roger
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.12366
  raw: '[[raw/papers/md/2026-classifier-context-rot-monitor-performance-degrades-with-context-length]]'
  source: https://arxiv.org/abs/2605.12366
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-classifier-context-rot-monitor-performance-degrades-with-context-length.md
raw_pdf: raw/papers/pdf/2026-classifier-context-rot-monitor-performance-degrades-with-context-length.pdf
read: false
slug: classifier-context-rot-monitor-performance-degrades-with-context-length
tags:
- type/paper
- status/stub
- long-context
- ai-safety
- monitoring
- evaluation
title: 'Classifier Context Rot: Monitor Performance Degrades with Context Length'
type: note
updated: '2026-05-18'
year: 2026
---

# Classifier Context Rot: Monitor Performance Degrades with Context Length

> *Sam Martin, Fabien Roger* — arXiv 2026

## TL;DR

Monitoring coding agents for dangerous behavior means classifying transcripts that often exceed 500K tokens, but prior agent-monitoring benchmarks rarely contain transcripts past 100K. When used as **classifiers**, current frontier models miss dangerous actions more often in longer transcripts: on a dataset requiring detection of subtly dangerous coding-agent actions, Opus 4.6, GPT 5.4, and Gemini 3.1 miss them **2×–30× more often** when they occur after 800K tokens of benign activity than in isolation. Partially mitigable with prompting (periodic reminders throughout the transcript) and possibly further with better post-training. Takeaway: monitor evaluations that ignore long-context degradation likely overestimate monitor performance. (Summary from abstract; note unread.)

## Abstract

Monitoring coding agents for dangerous behavior using language models requires classifying transcripts that often exceed 500K tokens, but prior agent monitoring benchmarks rarely contain transcripts longer than 100K tokens. We show that when used as classifiers, current frontier models fail to notice dangerous actions more often in longer transcripts. In particular, on a dataset that requires identifying when a coding agent takes a subtly dangerous action, Opus 4.6, GPT 5.4, and Gemini 3.1 miss these actions $2\times$ to $30\times$ more often when they occur after 800K tokens of benign activity than when they occur on their own. We also show that these weaknesses can be partially mitigated with prompting techniques such as periodic reminders throughout the transcript and may be mitigated further with better post-training. Monitor evaluations that do not consider long-context degradation are likely overestimating monitor performance.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.12366>
- PDF: [[raw/papers/pdf/2026-classifier-context-rot-monitor-performance-degrades-with-context-length.pdf]]
- Raw markdown: [[raw/papers/md/2026-classifier-context-rot-monitor-performance-degrades-with-context-length]]
