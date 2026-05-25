---
arxiv: '2605.23055'
authors:
- Changling Li
- Terry Jingchen Zhang
- Jie Zhang
- Zhijing Jin
- Sahar Abdelnabi
- Maksym Andriushchenko
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.23055
  raw: '[[raw/papers/md/2026-decomposing-and-measuring-evaluation-awareness]]'
  source: https://arxiv.org/abs/2605.23055
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-decomposing-and-measuring-evaluation-awareness.md
raw_pdf: raw/papers/pdf/2026-decomposing-and-measuring-evaluation-awareness.pdf
read: false
slug: decomposing-and-measuring-evaluation-awareness
tags:
- type/paper
- status/stub
title: Decomposing and Measuring Evaluation Awareness
type: note
updated: '2026-05-25'
year: 2026
---

# Decomposing and Measuring Evaluation Awareness

> *Changling Li, Terry Jingchen Zhang, Jie Zhang, Zhijing Jin, Sahar Abdelnabi, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Frontier language models sometimes recognize that they are being evaluated and adjust their behavior, undermining validity of benchmark results. Yet the field studies it without a shared foundation, conflating properties of the evaluation with properties of the model, and detection with behavioral response. We ground evaluation awareness in social psychology, decomposing it into an environment component (how recognizable the task is) and a model component that separates recognition from propensity to act on it. We operationalize the environment component through eight categorized trigger factors, such as placeholder entities and grading-style output formats, and study recognition and behavior through chain-of-thought monitoring. Across nine frontier models and four benchmarks, recognition rates depend on the specific pairing of model and benchmark rather than on either in isolation. Recognition rarely leads to behavioral change, and when it does, the direction depends on the type of evaluation perceived. Models are also more sensitive to safety than capability evaluations, placing safety benchmark validity at greater risk. To study which factors each model is sensitive to and how they interact, we propose \textbf{EvalAwareBench}, a factor-controlled benchmark of 100 paired safety-capability tasks where each of the eight factors can be independently toggled, varying evaluative signals while holding the underlying request fixed. Through EvalAwareBench, we find that no single factor uniformly affects all models, but stacking factors progressively raises evaluation awareness across all of them. Our framework and EvalAwareBench provide the tools to measure, attribute, and mitigate evaluation awareness, pointing to behavioral consistency under recognition as a promising path forward.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.23055>
- PDF: [[raw/papers/pdf/2026-decomposing-and-measuring-evaluation-awareness.pdf]]
- Raw markdown: [[raw/papers/md/2026-decomposing-and-measuring-evaluation-awareness]]
