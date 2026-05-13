---
arxiv: '2505.19641'
authors:
- Junteng Liu et al
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.19641
  raw: https://arxiv.org/pdf/2505.19641
  source: https://arxiv.org/abs/2505.19641
owner: blaz
read: false
slug: synlogic
tags:
- type/paper
- status/stub
- domain/reasoning
title: 'SynLogic: Synthesising Verifiable Reasoning Data at Scale for Reinforcement
  Learning'
type: note
updated: '2026-05-10'
year: 2025
---

# SynLogic: Synthesising Verifiable Reasoning Data at Scale for Reinforcement Learning

## Citation

- URL: https://arxiv.org/abs/2505.19641
- PDF: https://arxiv.org/pdf/2505.19641
- Authors: Junteng Liu et al.
- Year / venue: 2025 (arXiv submission May 2025).

## Core Claim

SynLogic synthesises diverse logical-reasoning tasks (Sudoku variants, game-24, cryptarithmetic, cipher decoding, and other puzzle families) whose correctness can be checked by **simple rule-based verifiers**. The pipeline scales to hundreds of thousands of instances with guaranteed ground-truth labels, enabling reinforcement learning with verifiable outcome rewards for logical reasoning.

## Key Paper Ideas

- **Rule-based checkers as the oracle.** For each task family, a short verifier script decides correctness — avoiding the LLM-judge reliability problem entirely. See [[concepts/verification-signals]] §Hard × Outcome (rule-based symbolic checker) and [[metrics/verification-accuracy]].
- **Task-family diversity.** Multiple distinct puzzle families cover a range of reasoning patterns (constraint satisfaction, arithmetic synthesis, substitution). The diversity is important for generalisation beyond any single puzzle type.
- **Direct RL compatibility.** Because ground-truth labels are free and deterministic, SynLogic data supports verifiable-reward RL (DeepSeek-R1-style) out-of-the-box — no reward model needed.
- **Held-out validation splits.** The paper reports performance using variance-reducing metrics such as *average@8* (average pass-rate over 8 samples).

## Why KB cites this

- [[concepts/reasoning-data-generation]] §Archetype 2 — tool-verified synthesis with rule-based checkers; also §Archetype 3 (the data is directly usable as RL training signal with verifiable reward).
- [[metrics/verification-accuracy]] — rule-based checkers are the $f_\text{check}$ implementation here.
- [[concepts/synthetic-data-formalism]] §"V in verification-heavy domains" — a clean instance where $V$ is deterministic and free.
- [[concepts/verification-signals]] §Hard × Outcome — rule-based symbolic checking over puzzle answers; the same cell that hosts proof-graph match (ProofWriter) and on-diff similarity (SWE-RL).

## Core Concepts

- Existing concepts: [[concepts/reasoning-data-generation]], [[concepts/verification-signals]], [[concepts/process-vs-outcome-reward]].
- Concepts to extract: none new — absorbed.

## Relevance To Poolside

Puzzle-style synthetic reasoning data with deterministic checkers is a cheap, high-fidelity source of RL training signal, especially when targeting systematic generalisation across constraint families. Worth a look if internal RL pipelines are bottlenecked on reward-model quality for non-code reasoning.

## Related Notes

- Concepts: [[concepts/reasoning-data-generation]], [[concepts/verification-signals]], [[concepts/process-vs-outcome-reward]].
- Metrics: [[metrics/verification-accuracy]].
- Papers: [[notes/papers/2021-proofwriter]] (lineage — rule-grounded reasoning data).

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration.
