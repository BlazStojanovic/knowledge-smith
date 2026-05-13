---
arxiv: '2012.13048'
authors:
- Oyvind Tafjord
- Bhavana Dalvi Mishra
- Peter Clark
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2012.13048
  raw: https://arxiv.org/pdf/2012.13048
  source: https://arxiv.org/abs/2012.13048
owner: blaz
read: false
slug: proofwriter
tags:
- type/paper
- status/stub
- domain/reasoning
title: 'ProofWriter: Generating Implications, Proofs, and Abductive Statements over
  Natural Language'
type: note
updated: '2026-05-10'
year: 2021
---

# ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language

## Citation

- URL: https://arxiv.org/abs/2012.13048
- PDF: https://arxiv.org/pdf/2012.13048
- Authors: Oyvind Tafjord, Bhavana Dalvi Mishra, Peter Clark.
- Year / venue: 2021 (ACL / Findings 2021); arXiv submission Dec 2020.

## Core Claim

ProofWriter is a benchmark and generation framework for natural-language deductive reasoning grounded in formal rules. Given a theory (facts + rules in controlled natural language) and a statement, models must predict entailment, produce a proof graph, and answer abductive queries. The paper introduces the **strict "Full Accuracy"** metric: the predicted proof graph must exactly match the gold proof; any mismatch zeroes the score.

## Key Paper Ideas

- **Rule-grounded deductive instances.** Theories are synthesised programmatically from a rule base (CWA, OWA, Open splits — closed-world, open-world, open-world with abduction). Validity is therefore decidable by symbolic check rather than LLM judge.
- **Strict proof-graph exact match.** The canonical outcome-level validity metric for formal-logic reasoning. See [[metrics/strict-proof-accuracy]] for the $\mathrm{Acc}_\text{proof}$ definition and the contrast with weaker outcome-only accuracy.
- **Full proof trees.** Models generate the proof graph end-to-end (not just the conclusion), enabling step-level auditability — a prerequisite for process-level faithfulness analysis (FaiRR, ReCEval).
- **Abductive queries.** Beyond entailment, the benchmark includes abduction: given a statement, produce the missing facts that would make the statement follow from the theory.

## Why KB cites this

- [[concepts/reasoning-data-generation]] §Archetype 2 — canonical tool-verified synthesis with a symbolic checker as $V$.
- [[metrics/strict-proof-accuracy]] — the Full Accuracy protocol is the $\mathrm{Acc}_\text{proof}$ definition.
- [[concepts/synthetic-data-formalism]] §"V in verification-heavy domains" — (S = rule-base, M = theory templates, G = program, f = proof-graph match, V = symbolic checker).
- [[maps/grounding/formal-spec]] — rule-grounded, deterministically checkable reasoning data.

## Core Concepts

- Existing concepts: [[concepts/reasoning-data-generation]], [[concepts/verification-signals]] (Hard × Outcome — rule-based symbolic checker over a deductive rule base).
- Concepts to extract: none new — absorbed.

## Relevance To Poolside

Rule-grounded deductive data provides the cleanest possible supervision signal (symbolic oracle, proof-level granularity). For Poolside pipelines that need high-reliability reasoning supervision, ProofWriter-style synthesis is a strong candidate when the target domain admits a rule-base encoding.

## Related Notes

- Concepts: [[concepts/reasoning-data-generation]], [[concepts/verification-signals]].
- Metrics: [[metrics/strict-proof-accuracy]].
- Papers: [[notes/papers/2022-fairr]] (extends with consistency + entailment accuracy triple), [[notes/papers/2025-synlogic]] (rule-based logic synthesis lineage).
- Maps: [[maps/grounding/formal-spec]].

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration.
