---
arxiv: '2407.01502'
authors:
- Kapoor*
- Stroebl*
- Siegel
- Nadgir
- Narayanan (Princeton University)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.01502
  raw: '[[raw/papers/md/2024-ai-agents-that-matter]]'
  source: https://arxiv.org/abs/2407.01502
owner: blaz
raw_pdf: raw/papers/pdf/2024-ai-agents-that-matter.pdf
read: false
slug: ai-agents-that-matter
tags:
- type/paper
- status/draft
- domain/evals
- domain/agents
- domain/code
- source/primary
title: AI Agents That Matter
type: note
updated: '2026-05-10'
year: 2024
---

# AI Agents That Matter

## Citation

- URL: https://arxiv.org/abs/2407.01502
- Authors: Kapoor*, Stroebl*, Siegel, Nadgir, Narayanan (Princeton University)
- Year / venue: 2024 / arXiv
- **Raw**: [[raw/papers/pdf/2024-ai-agents-that-matter]]

## Core Claim

Agent evaluations must be cost-controlled — without cost normalization, simple retry/escalation strategies match "SOTA" agent architectures on HumanEval at 2-50x lower cost, meaning reported accuracy gains may be scientifically meaningless. The paper identifies five systemic problems in agent evaluation: lack of cost control, no joint cost-accuracy optimization, conflation of model vs downstream evaluation, benchmark overfitting via shortcuts, and reproducibility failures.

## Key Paper Ideas

- **Cost-controlled evaluation is non-negotiable**: on HumanEval, a simple "warming" strategy (retry with increasing temperature) matches LATS/LDB/Reflexion at dramatically lower cost. Without cost control, the community cannot distinguish genuine advances from brute-force retrying.
- **Pareto curve reporting**: plot accuracy vs cost, identify Pareto frontier. An agent that is not on the frontier has not demonstrated progress.
- **Model evaluation vs downstream evaluation**: model eval uses compute (FLOPs, parameters) as cost axis and should not change over time; downstream eval uses dollar cost and captures real-world deployment economics. NovelQA conflates these — RAG costs 20x less in realistic sequential querying but the benchmark obscures this.
- **Agent generality taxonomy**: four levels (distribution-specific / task-specific / domain-general / fully general), each requiring different holdout strategies. Only 1/8 domain-general and 0/2 fully-general benchmarks have appropriate holdouts.
- **Reproducibility failures are agent-specific**: eval scripts assume specific agent designs, LLM-to-agent benchmark repurposing introduces bugs, high cost prevents confidence intervals ($8K per SWE-bench run), external dependencies create non-determinism (rate limits, API changes).

## Methodology

**Simple baselines on HumanEval**: Retry (up to 5 calls at T=0), Warming (retry with T=0→0.5), Escalation (start Llama-3 8B, escalate through GPT-3.5/Llama-3 70B/GPT-4). Compare to LATS, LDB, Reflexion.

**Joint optimization**: modify DSPy on HotPotQA with Optuna search over temperature, few-shot count, example selection, formatting instructions. GPT-3.5: 53% lower cost at same accuracy; Llama-3-70B: 41% lower cost.

**Holdout audit**: survey 17 agent benchmarks, classify by generality level, check holdout appropriateness. 7/17 have no holdout at all.

**Reproducibility audit**: attempt to reproduce reported results for multiple agents on HumanEval and WebArena. Many reported scores exceed max of 5 reproduction runs.

## Key Results

| Finding | Value |
|---|---|
| HumanEval: Warming vs LATS/LDB/Reflexion | Comparable accuracy, 2-50x lower cost |
| HotPotQA joint opt: GPT-3.5 cost reduction | 53% at same accuracy |
| HotPotQA joint opt: Llama-3-70B cost reduction | 41% at same accuracy |
| Agent benchmarks with no holdout | 7/17 |
| Domain-general benchmarks with appropriate holdout | 1/8 |
| Fully-general benchmarks with appropriate holdout | 0/2 |
| SWE-bench: cost per single evaluation run | >$8,000 |

**WebArena case study**: STeP achieves 35.8% (>2x baseline) by hardcoding task-specific policies — brittle to layout drift, exploiting lack of holdout tasks.

## Core Concepts

- [[concepts/agentic-evaluation-methodology]] — this paper defines the key problems and design patterns
- [[concepts/evaluation-variance]] — agent evaluation has unique variance sources (environmental, cost-scaling)
- [[concepts/benchmark-saturation]] — the overfitting/shortcut analysis is about a form of artificial saturation

## Relevance To Poolside

Directly relevant to Harbor-based agent evals (SWE-bench Verified, Terminal Bench, agent sanity check). The cost-control argument applies to all agentic evals where iterative LLM calls are possible. The generality taxonomy maps to Poolside's eval categorization (distribution-specific → per-task evals, domain-general → AGENT_DEFAULT group). The reproducibility concerns (environment drift, rate limits) affect Harbor sandbox evaluations.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- Does the warming baseline result extend to SWE-bench (harder than HumanEval)?
- How to set the cost budget for agentic evals at Poolside — fixed dollar per instance? Fixed number of LLM calls?
- The joint optimization finding suggests DSPy-style auto-tuning should be standard — is this being done for Harbor runs?

## Related Notes

- Evals: [[evals/swe-bench-verified-harbor]], [[evals/terminal-bench-v2-harbor]], [[evals/agent-sanity-check]], [[evals/humaneval]]
- Papers: [[notes/papers/2023-judging-llm-as-a-judge]]
- Concepts: [[concepts/agentic-evaluation-methodology]]

## Caveats

- HumanEval is relatively easy — the warming-matches-SOTA finding may not hold for harder benchmarks like SWE-bench
- Joint optimization adds upfront cost (break-even at ~1,350 tasks on HotPotQA)
- Dollar costs are volatile — results change as API pricing changes
- The paper acknowledges System 2 techniques may prove valuable on harder tasks
