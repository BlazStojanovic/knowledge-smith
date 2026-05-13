---
arxiv: '2203.10261'
authors:
- Soumya Sanyal
- Harman Singh
- Xiang Ren
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2203.10261
  raw: https://arxiv.org/pdf/2203.10261
  source: https://arxiv.org/abs/2203.10261
owner: blaz
read: false
slug: fairr
tags:
- type/paper
- status/stub
- domain/reasoning
title: 'FaiRR: Faithful and Robust Deductive Reasoning over Natural Language'
type: note
updated: '2026-05-10'
year: 2022
---

# FaiRR: Faithful and Robust Deductive Reasoning over Natural Language

## Citation

- URL: https://arxiv.org/abs/2203.10261
- PDF: https://arxiv.org/pdf/2203.10261
- Authors: Soumya Sanyal, Harman Singh, Xiang Ren.
- Year / venue: 2022 (ACL 2022); arXiv submission March 2022.

## Core Claim

FaiRR is a deductive-reasoning model evaluated under a three-part protocol: entailment accuracy, strict proof accuracy ([[metrics/strict-proof-accuracy]] $\mathrm{Acc}_\text{proof}$), and **perturbation-equivalence consistency** ($C(T, s)$). The consistency metric — invariance of the model's answer under meaning-preserving perturbations of the theory $T$ and statement $s$ — is FaiRR's main contribution to the evaluation toolkit.

## Key Paper Ideas

- **Consistency as a robustness metric.** For each problem, construct an equivalence set of perturbed inputs (surface rewrites, operand substitutions, logically equivalent rule reorderings). The model's consistency is the fraction of perturbations on which it returns the same answer. See [[metrics/strict-proof-accuracy]] §FAIRR consistency.
- **Triple reporting.** Accuracy + $\mathrm{Acc}_\text{proof}$ + $C$ captures three distinct failure modes — right-for-wrong-reasons, correct-but-non-robust, consistent-but-wrong. Reporting only accuracy obscures the last two.
- **Faithful rule selection.** FaiRR's architecture selects rules and facts explicitly at each step, exposing the reasoning process to auditing — a structural faithfulness guarantee that complements the metric-level perturbation consistency.

## Why KB cites this

- [[metrics/strict-proof-accuracy]] — source of FAIRR consistency and of the three-part reporting convention.
- [[concepts/trustworthiness-taxonomy]] §Robustness — the perturbation-consistency measure is the formal-logic specialisation of the robustness pillar.
- [[concepts/reasoning-data-generation]] §Archetype 2 — benchmark-grade evaluation of pipelines that use rule-based reasoning data.

## Core Concepts

- Existing concepts: [[concepts/trustworthiness-taxonomy]], [[concepts/verification-signals]].
- Concepts to extract: none new — absorbed under trustworthiness-taxonomy §Robustness.

## Relevance To Poolside

Perturbation-consistency audits are directly transposable to any reasoning domain with a meaningful equivalence class of inputs — program refactoring preserves semantics, variable renaming preserves behaviour, type-equivalent API calls preserve meaning. A FaiRR-style $C$ on a Poolside code-reasoning pipeline would surface surface-feature-exploiting failures that accuracy metrics miss.

## Related Notes

- Concepts: [[concepts/trustworthiness-taxonomy]], [[concepts/reasoning-data-generation]].
- Metrics: [[metrics/strict-proof-accuracy]].
- Papers: [[notes/papers/2021-proofwriter]] (benchmark this builds on).

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration.
