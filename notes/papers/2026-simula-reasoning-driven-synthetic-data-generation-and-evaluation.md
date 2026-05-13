---
aliases:
- 2026-simula
- simula
arxiv: '2603.29791'
authors:
- Tim R. Davidson
- Benoit Seguin
- Enrico Bacis
- Cesar Ilharco
- Hamza Harkous
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.29791
  raw: '[[raw/papers/md/2026-simula-reasoning-driven-synthetic-data-generation-and-evaluation]]'
  source: https://arxiv.org/abs/2603.29791
owner: blaz
raw_pdf: raw/papers/pdf/2026-simula-reasoning-driven-synthetic-data-generation-and-evaluation.pdf
read: true
slug: simula-reasoning-driven-synthetic-data-generation-and-evaluation
tags:
- type/paper
- source/primary
- status/verified
- confidential/public-source
- domain/general
- domain/llm
- domain/synth-data
- domain/data-mix
- domain/evals
- domain/agents
- domain/models
- stage/sft
title: 'Simula: Reasoning-Driven Synthetic Data Generation and Evaluation'
type: note
updated: '2026-05-10'
year: 2026
---

# Simula: Reasoning-Driven Synthetic Data Generation and Evaluation

- **arXiv**: [2603.29791](https://arxiv.org/abs/2603.29791)
- **Authors**: Tim R. Davidson, Benoit Seguin, Enrico Bacis, Cesar Ilharco, Hamza Harkous
- **Year / venue**: 2026-03-31 (arXiv preprint; OpenReview submission)
- **Raw**: [[raw/papers/pdf/2026-simula-reasoning-driven-synthetic-data-generation-and-evaluation]]
- **Grounding axis**: [[maps/grounding/knowledge-structure]] (Axis 8a — self-expanding taxonomy). The paper is explicitly **seedless**; the agentically-constructed taxonomy is the sole scaffolding.
- **Output shape**: 512k synthetic (instance) samples per task after deduplication and critique filtering.
- **Filter / verification**: M3 point-wise critique + "double-critic" (independent correctness / incorrectness scores) to reduce sycophancy.
- **Training stage**: Post-training (LoRA fine-tune on Gemma 3 4B student). Paper does **not** validate at pretrain scale.

## Method

Three-stage pipeline, all M3-driven (multimodal model used as generator + critic):

1. **Taxonomy construction.** M3 iteratively proposes child nodes given a parent node, then re-enters as a critic to refine. Optional planning step enforces consistent granularity across parallel branches. Produces a hierarchical decomposition of "factors of variation" for the target domain.
2. **Taxonomic sampling + meta-prompting.** Node-subsets sampled under user-defined strategies (weighted grouping). M3 converts sampled requirement-sets into natural-language "meta prompts" for generation.
3. **Agentic refinement.** Generated candidates go through point-wise M3 critique. Complexification is applied to a fraction (default c=0.5) of prompts to increase difficulty.

Teacher / generator: **Gemini 2.5 Flash** (non-thinking). Student: **Gemma 3 4B**. Teacher and student are always distinct.

## Key results

- Taxonomy quality: **0.74 completeness, 0.75 soundness** on grounded taxonomies; **0.78 / 0.97** on conceptual ones (vs 0-shot baselines of 0.52/0.70 and 0.50/0.97).
- Downstream: "full Simula system is almost always the dominant strategy across all datasets and data sizes" on CTI-MCQ, CTI-RCM, LEXam, GSM8k, and Global MMLU (Korean/Nepali subsets).
- Complexity sweep: +10 % accuracy on GSM8k at 64k samples between low- and high-complexity splits.
- Double-critic: in controlled settings μ_critic > μ_gen consistently; empirical setting shows smaller but positive lift.
- Critic rejection rates: 2 % (CTI-MCQ), 9 % (CTI-RCM, GSM8k), 61 % (LEXam with weak teacher).

## Critique

*Our synthesis — distinct from the paper's claims.*

**Raised by Blaz:**

- **Taxonomy coverage of the input space is fundamentally limited.** The "Level Ratio Coverage" metric is *within-taxonomy* — it measures completeness against the M3-generated structure itself, not the true target distribution. Deeper taxonomies hit combinatorial leaf sparsity; shallow ones miss long-tail regions. The 0.74 completeness ceiling is set by the taxonomy, not by the input space.
- **Skew toward distillation.** Framing aside, Gemini 2.5 Flash → Gemma 3 4B is a classic teacher → student setup. "Reasoning-first" doesn't change the information-theoretic fact that the student inherits the teacher's distribution.
- **No teacher = student ablation.** Missing: Gemma 3 4B generating for itself (self-alignment regime — see [[concepts/self-alignment-vs-distillation]]). Without this, the reported gains cannot be decomposed into "reasoning-driven taxonomy helps" vs. "bigger teacher always helps." This is load-bearing for the headline claim.

**Additional observations:**

- **Weak control arm.** No baseline of document-grounded rephrasing (WRAP / [[notes/papers/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale]] / FinePhrase-style) on the same target tasks, so the added complexity of the taxonomy pipeline is not isolated.
- **Sycophancy mitigation is a patch.** The double-critic reduces a known LLM-judge pathology but doesn't escape it — the critic is still in the generator's family.
- **Scale mismatch.** 512 k samples / task is post-training scale. The paper's framing as general "data generation" over-claims against pretraining regimes where taxonomies become more brittle.

## Notes

- Fits squarely in the 2024–26 **self-expanding taxonomy** pattern ([[maps/grounding/knowledge-structure]] sub-pillar 8a) — alongside Explore-Instruct [2310.09168](https://arxiv.org/abs/2310.09168), CodecLM [2404.05875](https://arxiv.org/abs/2404.05875), Instruct-SkillMix [2408.14774](https://arxiv.org/abs/2408.14774). The agentic critic loop is the distinguishing engineering detail.
- The complexification operator (c=0.5) is Evol-Instruct-adjacent ([2304.12244](https://arxiv.org/abs/2304.12244)) but applied to the *prompt* after taxonomy sampling, not to a seed instruction.
- OpenReview PDF: [openreview.net/pdf?id=NALsdGEPhB](https://openreview.net/pdf?id=NALsdGEPhB).

## Related notes

- [[concepts/rephrasal-operations]]
- [[concepts/synthetic-data-formalism]]

## Open threads

- What is the true-distribution coverage of an M3-generated taxonomy at depth 3–5? No metric in the paper addresses this directly.
- Would Simula still dominate if paired with a taxonomy **extracted from organic data** (InsTag-style) rather than generated from first principles?
- Teacher = student version of the full pipeline — is the agentic loop still a net positive, or does it just distribute frontier-model capability?
