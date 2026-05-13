---
arxiv: '2402.16352'
authors:
- Zimu Lu
- Aojun Zhou
- Houxing Ren
- Ke Wang
- Weikang Shi
- Junting Pan
- Mingjie Zhan
- Hongsheng Li
created: 2026-04-29
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2402.16352
  raw: https://arxiv.org/pdf/2402.16352
  source: https://arxiv.org/abs/2402.16352
owner: blaz
read: false
slug: mathgenie-generating-synthetic-data-with-question-back-translation-for-enhancing-mathematical-reasoning-of-llms
tags:
- type/paper
- status/stub
- source/primary
- domain/synth-data
- domain/math
- domain/reasoning
title: 'MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing
  Mathematical Reasoning of LLMs'
type: note
updated: '2026-05-10'
year: 2024
---

# MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs

## Citation

- URL: https://arxiv.org/abs/2402.16352
- PDF: https://arxiv.org/pdf/2402.16352
- Authors: Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li
- Year / venue: ACL 2024 (Long), arXiv preprint Feb 2024
- ACL Anthology: https://aclanthology.org/2024.acl-long.151/

## Core Claim

Augments a small seed problem-solution math dataset by (i) mutating ground-truth solutions, (ii) training a back-translation model that maps an augmented solution back to a new natural-language question, and (iii) generating code-integrated solutions for the new questions with rationale-based correctness verification. The resulting MathGenieLM family (7B–70B) reaches 87.7% GSM8K / 55.7% MATH on InternLM2.

## Key Paper Ideas

- **Question back-translation** as a synthetic-data primitive: solution → question, inverting the conventional question → solution direction.
- Rationale-based verification of code-integrated solutions to filter the synthesised pairs.

## Methodology

- Seed = small math problem-solution corpus.
- Step 1: augment ground-truth solutions.
- Step 2: train a solution → question back-translator on (solution, question) pairs.
- Step 3: generate code-integrated solutions for the back-translated questions.
- Step 4: rationale-based verification keeps consistent (question, solution) pairs.

## Experiments

- Train 7B–70B base models on the curated MathGenie data; evaluate on five math reasoning benchmarks (GSM8K, MATH, +three).

## Key Results

- MathGenieLM-InternLM2: GSM8K 87.7%, MATH 55.7%; SOTA among open-source models at the time across five math reasoning benchmarks.

## Core Concepts

- Existing concepts: [[concepts/rephrasal-operations]] (archetype #5 cross-modal projection — solution↔question is a back-translation special case), [[concepts/reasoning-data-generation]], [[concepts/synthetic-data-formalism]].
- Concepts to extract: —

## Relevance To Poolside

*Stub.* Back-translation is a generally useful inversion pattern; the analogue for code is solution → problem (cf. Magicoder OSS-Instruct, InverseCoder). Worth comparing to existing code back-translation pipelines on signal quality and verification cost.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- 

## Related Notes

- Concepts: [[concepts/rephrasal-operations]], [[concepts/reasoning-data-generation]]

## Caveats

- Stub note; abstract-level only.
