---
arxiv: '2504.12491'
authors:
- Hansi Zeng
- Kai Hui
- Honglei Zhuang
- Zhen Qin
- Zhenrui Yue
- Hamed Zamani
- Dana Alon
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.12491
  raw: '[[raw/papers/md/2025-pretraining-indicators-predict-finetuning]]'
  source: https://arxiv.org/abs/2504.12491
owner: blaz
raw_pdf: raw/papers/pdf/2025-pretraining-indicators-predict-finetuning.pdf
read: false
slug: pretraining-indicators-predict-finetuning
tags:
- type/paper
- status/stub
- domain/evals
- domain/pretraining
title: Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?
type: note
updated: '2026-05-10'
year: 2025
---

# Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?

## Citation

- URL: https://arxiv.org/abs/2504.12491
- PDF: https://arxiv.org/pdf/2504.12491
- Authors: Hansi Zeng, Kai Hui, Honglei Zhuang, Zhen Qin, Zhenrui Yue, Hamed Zamani, Dana Alon
- Year / venue: 2025 (arXiv, April 2025)

## Core Claim

Standard perplexity measured at pre-training time is a **misleading predictor** of downstream fine-tuned performance. Alternative proxy metrics proposed by the authors reduce cross-dataset prediction error by over 50% in a controlled study across 50B-parameter model variants.

## Key Paper Ideas

- **Perplexity is unreliable.** Low pre-training PPL does not reliably rank models by post-fine-tuning benchmark score; a model with higher PPL can fine-tune to better performance.
- **Alternative proxies.** The paper proposes alternative metrics (not yet read in detail) that correlate more strongly with fine-tuning outcomes. Needs deeper read to characterise the metrics.
- **Controlled study.** 50B-parameter models with different pre-training setups evaluated across multiple downstream tasks — provides empirical grounding rather than a single data point.
- **Practical relevance.** Directly addresses the question: "which pre-training run should we continue?" — useful for data-curation decisions.

## Relevance To Poolside

Directly relevant to the model evaluation and data-selection work: if pre-training loss / PPL is not a reliable proxy for downstream quality, what should we be tracking? Informs how to instrument mid-training evaluation.

## Related Notes

- Concepts: [[concepts/evaluation-targets]]
- Maps: [[maps/evaluation/landscape]], [[maps/model-evaluation/code-eval-paradigms]]
- Metrics: [[metrics/verification-accuracy]]

## Caveats

Stub — created 2026-04-28. Key metrics in the paper not yet read; "alternative proxies" description needs updating after reading.
