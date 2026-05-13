---
arxiv: '2406.01006'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2406.01006
  raw: null
  source: https://arxiv.org/abs/2406.01006
owner: blaz
raw_pdf: raw/papers/pdf/2024-semcoder.pdf
read: false
slug: semcoder
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- stage/sft
title: SemCoder
type: note
updated: '2026-05-10'
year: 2024
---

# SemCoder

- **arXiv**: [2406.01006](https://arxiv.org/abs/2406.01006)
- **Authors / affiliation**: Yangruibo Ding et al. (ARiSE Lab, Columbia)
- **Year / venue**: 2024, NeurIPS 2024
- **Raw**: [[raw/papers/pdf/2024-semcoder]]
- **Grounding axis**: [[maps/grounding/structured-knowledge]] (cross-links [[maps/grounding/execution-traces]])
- **Output shape**: (code, step-by-step semantic "monologue") SFT pairs
- **Filter / verification**: Execution-verified traces (executable Python corpus "PyX" with tests)
- **Training stage**: SFT

## Method

Trains a Code LLM to reason about **comprehensive semantics**: high-level functional description, local execution effect of each statement, overall input / output behaviour. Two trace directions:

- **Forward monologue** — given source code and inputs, the model verbally simulates execution, explaining each line's impact, executed lines, variable state changes, and final output.
- **Backward monologue** — given the final output, the model reasons abstractly about possible prior states.

Training data: **PyX**, a clean Python corpus of fully executable samples paired with functional descriptions and test cases; monologue traces are produced and validated against real execution.

## Key result

SemCoder (6.7B parameters, based on DeepSeek-Coder 6.7B lineage) reported:

- HumanEval: **79.3%** (vs. GPT-3.5-turbo reported at 76.8%)
- CRUXEval-I: **63.6%** (vs. GPT-3.5-turbo reported at 50.3%)
- CRUXEval-O: **63.9%** (vs. GPT-3.5-turbo reported at 59.0%)

The CRUXEval improvements are the more striking — they measure reasoning about execution behaviour rather than functional correctness.

## Notes

- Unique among code SFT recipes in that the *reasoning trace* is directly grounded in structured execution semantics, not just post-hoc verified.
- Spans two axes: static structure enters via line-by-line effect prediction; executed state enters via forward simulation.
- Code: [github.com/ARiSE-Lab/SemCoder](https://github.com/ARiSE-Lab/SemCoder). See also [[maps/grounding/structured-knowledge]] §Sub-patterns — execution-static fusion.
