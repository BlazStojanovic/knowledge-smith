---
arxiv: '2403.09032'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2403.09032
  raw: null
  source: https://arxiv.org/abs/2403.09032
owner: blaz
raw_pdf: raw/papers/pdf/2024-codeultrafeedback.pdf
read: false
slug: codeultrafeedback
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- stage/dpo
title: CodeUltraFeedback
type: note
updated: '2026-05-10'
year: 2024
---

# CodeUltraFeedback

- **arXiv**: [2403.09032](https://arxiv.org/abs/2403.09032)
- **Authors / affiliation**: Martin Weyssow et al.
- **Year**: 2024 (TOSEM 2025)
- **Raw**: [[raw/papers/pdf/2024-codeultrafeedback]]
- **Grounding axis**: [[maps/grounding/execution-traces]] (cross-links [[maps/grounding/real-code-anchor]] via instruction anchor)
- **Output shape**: Preference pair for DPO / RLAIF
- **Filter / verification**: LLM-as-judge (GPT-3.5) across 5 coding-preference dimensions
- **Training stage**: Preference tuning (DPO)

## Method

10,000 complex coding instructions. For each: 14 diverse LLMs generate responses. GPT-3.5 is used as an LLM-as-judge to annotate responses against **five coding preferences**:

1. Instruction-following
2. Code explanation
3. Code complexity and efficiency
4. Code readability
5. Coding style

Judged scores yield preference pairs suitable for DPO. The paper also releases **CODAL-Bench**, a companion benchmark for evaluating alignment to these preferences.

## Key result

CodeLlama-7B-Instruct, aligned via RLAIF + DPO on CodeUltraFeedback, is reported by the paper to outperform 34B LLMs on CODAL-Bench. GPT-3.5 / GPT-4 responses are generally preferred over open-weight responses — a signal that alignment gaps between closed and open weights are meaningful on these axes.

## Notes

- Filter is LLM-as-judge; execution is not the primary signal. Some dimensions (e.g. "complexity and efficiency") would benefit from runtime measurement — open direction.
- Code / data: [github.com/martin-wey/CodeUltraFeedback](https://github.com/martin-wey/CodeUltraFeedback).
- See [[concepts/verification-signals]] §LLM-as-judge for category placement.
