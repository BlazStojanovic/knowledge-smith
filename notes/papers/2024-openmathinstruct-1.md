---
arxiv: '2402.10176'
authors:
- Shubham Toshniwal
- Ivan Moshkov
- Sean Narenthiran
- Daria Gitman
- Fei Jia
- Igor Gitman (NVIDIA)
created: 2026-04-23
kind: paper
links:
  code: https://github.com/NVIDIA/NeMo-Skills
  paper: https://arxiv.org/abs/2402.10176
  raw: https://arxiv.org/pdf/2402.10176
  source: https://arxiv.org/abs/2402.10176
owner: blaz
read: false
slug: openmathinstruct-1
tags:
- type/paper
- status/stub
- domain/reasoning
- domain/math
title: 'OpenMathInstruct-1: A 1.8M Math Instruction Tuning Dataset'
type: note
updated: '2026-05-10'
year: 2024
---

# OpenMathInstruct-1: A 1.8M Math Instruction Tuning Dataset

## Citation

- URL: https://arxiv.org/abs/2402.10176
- PDF: https://arxiv.org/pdf/2402.10176
- Authors: Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, Igor Gitman (NVIDIA).
- Year / venue: 2024 (arXiv submission Feb 2024).

## Core Claim

OpenMathInstruct-1 synthesises a 1.8M-example math SFT dataset in which solutions are represented in a **code-interpreter style** — natural-language reasoning interleaved with executable Python blocks. Correctness is enforced by retaining only solutions whose computations execute and whose final answer matches the ground truth. The paper also introduces **training-set coverage** (pass@k over $N$ sampled solutions) as a dataset-quality metric.

## Key Paper Ideas

- **Code-interpreter-style rationales.** Bridges natural-language and executable reasoning: steps that can be checked are checked. A process-level signal is implicitly available at each Python block. See [[concepts/process-vs-outcome-reward]] (the outcome oracle is the final answer; the Python blocks add a partial process signal).
- **Execution-filtered rejection sampling.** Solutions that fail to execute or produce the wrong final answer are discarded. $f_\text{check}$ = (executes $\land$ answer-matches).
- **Training-set coverage as a metric.** For each problem, sample $k$ candidate solutions; coverage = fraction of problems with $\geq 1$ correct solution. A $k$-augmented $\mathrm{Acc}_\text{verify}$; see [[metrics/verification-accuracy]] §"Training-set coverage".
- **Size + open release.** 1.8M examples, released under permissive license; the scale + licensing positions it as a baseline for math-SFT corpora.

## Why KB cites this

- [[concepts/reasoning-data-generation]] §Archetype 2 — canonical tool-verified synthesis with executable oracle.
- [[metrics/verification-accuracy]] — the rejection filter is an $\mathrm{Acc}_\text{verify}$ application with an executor as $f_\text{check}$; the training-set-coverage metric is a $k$-augmented variant.
- [[concepts/synthetic-data-formalism]] §"V in verification-heavy domains" — (S = math seeds, M = ∅, G = LLM with code interpreter, f = execution filter, V = executor).

## Core Concepts

- Existing concepts: [[concepts/reasoning-data-generation]], [[concepts/process-vs-outcome-reward]], [[concepts/verification-signals]] (Hard × Outcome — execution + answer-match; with implicit Hard × Process at each Python block).
- Concepts to extract: none new — absorbed.

## Relevance To Poolside

Code-interpreter-style synthesis is a natural pattern for any reasoning domain with a cheap executor. The training-set-coverage metric is also directly transposable to Poolside's internal math / code synthesis pipelines for dataset-quality audit.

## Related Notes

- Concepts: [[concepts/reasoning-data-generation]], [[concepts/process-vs-outcome-reward]], [[concepts/verification-signals]].
- Metrics: [[metrics/verification-accuracy]].

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration.
