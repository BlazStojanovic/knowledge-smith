---
arxiv: '2304.10703'
authors:
- Archiki Prasad
- Swarnadeep Saha
- Xiang Zhou
- Mohit Bansal
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2304.10703
  raw: https://arxiv.org/pdf/2304.10703
  source: https://arxiv.org/abs/2304.10703
owner: blaz
read: false
slug: receval
tags:
- type/paper
- status/stub
- domain/reasoning
- domain/evals
title: 'ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness'
type: note
updated: '2026-05-10'
year: 2023
---

# ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness

## Citation

- URL: https://arxiv.org/abs/2304.10703
- PDF: https://arxiv.org/pdf/2304.10703
- Authors: Archiki Prasad, Swarnadeep Saha, Xiang Zhou, Mohit Bansal.
- Year / venue: 2023 (EMNLP 2023); arXiv submission April 2023.

## Core Claim

ReCEval is a reference-free framework for evaluating reasoning chains along two axes: **correctness** (each step follows from prior steps via textual entailment) and **informativeness** (each step contributes new information toward the conclusion). The framework operationalises step-level evaluation of CoT without requiring a gold reasoning trace, using NLI scorers for both axes.

## Key Paper Ideas

- **Correctness as inter-step entailment.** Each step $c_t$ is scored by whether it is entailed by the concatenation of preceding steps $c_{<t}$. Corresponds to $\mathrm{Align}_\text{entail}$ in [[metrics/step-validity-rate]].
- **Informativeness orthogonal to correctness.** A valid step can be uninformative (restating earlier content, adding a redundant deduction). Informativeness captures this via the NLI score of the step against the statement-to-prove, penalising redundant entailments.
- **Reference-free evaluation.** No gold chain required — both metrics are computed from the chain alone + an NLI model. Makes the framework applicable at scale.
- **Soft PRM without trained labels.** ReCEval approximates process-level reward modelling by using NLI scores instead of step-level human labels, avoiding PRM800K-style annotation cost.

## Why KB cites this

- [[metrics/step-validity-rate]] — ReCEval correctness is the canonical reference for $\mathrm{Val}_\text{step}$ / $\mathrm{Align}_\text{entail}$ with NLI as $f_\text{exec}$. Informativeness is the complementary axis that this note flags but does not separately measure.
- [[concepts/process-vs-outcome-reward]] — a soft-PRM instance where the process oracle is an NLI model rather than a trained PRM.
- [[concepts/reasoning-chain-judges]] — ReCEval is the structural template for a reasoning-chain judge that scores processes rather than outcomes.

## Core Concepts

- Existing concepts: [[concepts/process-vs-outcome-reward]], [[concepts/reasoning-chain-judges]], [[concepts/verification-signals]] (Soft × Process — NLI step-entailment).
- Concepts to extract: informativeness as an orthogonal axis to validity is worth a future concept-note pass if more papers adopt it; currently only ReCEval-instantiated.

## Relevance To Poolside

Reference-free chain-level evaluation is directly useful for auditing internal reasoning-data pipelines where gold chains don't exist. ReCEval's NLI-based correctness is also a cheap baseline verifier for open-ended reasoning where no executable oracle is available.

## Related Notes

- Concepts: [[concepts/process-vs-outcome-reward]], [[concepts/reasoning-chain-judges]], [[concepts/verification-signals]].
- Metrics: [[metrics/step-validity-rate]].

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration.
