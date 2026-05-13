---
arxiv: '2601.03448'
authors:
- Atsuki Yamaguchi
- Maggie Mi
- Nikolaos Aletras
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.03448
  raw: '[[raw/papers/md/2026-enhancing-linguistic-competence-l2t]]'
  source: https://arxiv.org/abs/2601.03448
owner: blaz
raw_pdf: raw/papers/pdf/2026-enhancing-linguistic-competence-l2t.pdf
read: false
slug: enhancing-linguistic-competence-l2t
tags:
- type/paper
- status/stub
- domain/synth-data
- domain/pretraining
title: Enhancing Linguistic Competence of Language Models through Pre-training with
  Language Learning Tasks
type: note
updated: '2026-05-10'
year: 2026
---

# Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks

## Citation

- URL: https://arxiv.org/abs/2601.03448
- PDF: https://arxiv.org/pdf/2601.03448
- Authors: Atsuki Yamaguchi, Maggie Mi, Nikolaos Aletras
- Year / venue: 2026 (arXiv, January 2026)

## Core Claim

L2T (Language Learning Tasks) mixes structured input-output pairs derived from raw text alongside standard next-token prediction during pre-training, inspired by human language acquisition. Models trained on mixed raw text + L2T data score higher on linguistic benchmarks while maintaining competitive general-reasoning performance.

## Key Paper Ideas

- **L2T framework.** Converts raw text into structured tasks (fill-in-the-blank, dependency parsing, semantic role labelling, etc.) to provide explicit linguistic stimulation during pre-training — similar in spirit to [[notes/papers/2024-instruction-pre-training]] but focused on linguistic rather than task-instruction supervision.
- **Inspired by human language acquisition.** Frames the approach as providing structured practice akin to language-learning exercises, not just next-token exposure.
- **Linguistic benchmark gains.** Improved performance on linguistic competence tests (likely morphology, syntax, semantics benchmarks) without sacrificing general reasoning.
- **Mix ratio matters.** The right fraction of L2T data relative to raw text is likely a key hyperparameter (needs deeper read).

## Relevance To Poolside

Extends the space of structured pre-training data beyond instruction-style pairs; worth tracking as a point in the design space of how to enrich pre-training corpora with task-structured signal.

## Related Notes

- Papers: [[notes/papers/2024-instruction-pre-training]]
- Concepts: [[concepts/synthetic-data-formalism]], [[concepts/reasoning-data-generation]]

## Caveats

Stub — created 2026-04-28.
