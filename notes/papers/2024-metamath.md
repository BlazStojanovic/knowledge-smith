---
arxiv: '2309.12284'
authors:
- Longhui Yu
- Weisen Jiang
- Han Shi
- Jincheng Yu
- Zhengying Liu
- Yu Zhang
- James T. Kwok
- Zhenguo Li
- Adrian Weller
- Weiyang Liu
created: 2026-04-23
kind: paper
links:
  code: https://github.com/meta-math/MetaMath
  paper: https://arxiv.org/abs/2309.12284
  raw: https://arxiv.org/pdf/2309.12284
  source: https://arxiv.org/abs/2309.12284
owner: blaz
read: false
slug: metamath
tags:
- type/paper
- status/stub
- domain/reasoning
- domain/math
title: 'MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models'
type: note
updated: '2026-05-10'
year: 2024
---

# MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models

## Citation

- URL: https://arxiv.org/abs/2309.12284
- PDF: https://arxiv.org/pdf/2309.12284
- Authors: Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu
- Year / venue: 2024 (ICLR 2024); arXiv submission Sep 2023.

## Core Claim

MetaMath scales up math-reasoning SFT corpora by (a) bootstrapping question variations from GSM8K and MATH seeds using rephrasing + forward / backward / self-verification reasoning templates, and (b) filtering generated (question, rationale, answer) triples by rejection-sampling on final-answer correctness. Models fine-tuned on MetaMathQA improve substantially on math benchmarks without changing the base model.

## Key Paper Ideas

- **Answer augmentation via rejection sampling.** Multiple reasoning paths are sampled; only those yielding the correct final answer are retained. This is an **outcome-level** filter — see [[concepts/process-vs-outcome-reward]] and [[metrics/verification-accuracy]].
- **Question bootstrapping.** Seed questions are mutated (rephrase, self-verify, backward-reasoning) to expand the training distribution beyond the seed size.
- **Pure-SFT regime.** No reinforcement learning; all gains come from the synthetic SFT corpus.

## Why KB cites this

- [[concepts/reasoning-data-generation]] §Archetype 1 — heuristic evolution (question bootstrapping) + §Archetype 2 (outcome-oracle filter via rejection sampling). MetaMath is the canonical example of a pipeline that straddles the two archetypes.
- [[metrics/verification-accuracy]] — the rejection-sampling filter is an $\mathrm{Acc}_\text{verify}$ application with the math-answer checker as $f_\text{check}$.
- [[concepts/synthetic-data-formalism]] §"V in verification-heavy domains" — (S = GSM8K/MATH seeds, M = mutation operators, G = LLM, f = answer-match, V = numerical checker).

## Core Concepts

- Existing concepts: [[concepts/reasoning-data-generation]], [[concepts/process-vs-outcome-reward]], [[concepts/verification-signals]] (Hard × Outcome — answer-match filter).
- Concepts to extract: none new — the main ideas are absorbed into existing concept notes.

## Relevance To Poolside

Outcome-oracle-filtered math SFT corpora — the dominant internal pattern. This note is a pointer for the Auditor §3 integration.

## Related Notes

- Concepts: [[concepts/reasoning-data-generation]], [[concepts/process-vs-outcome-reward]], [[concepts/verification-signals]].
- Metrics: [[metrics/verification-accuracy]].

## Caveats

- Stub — created in Phase 4b of the Auditor §3 integration. Not yet enriched via the paper-reading skill.
