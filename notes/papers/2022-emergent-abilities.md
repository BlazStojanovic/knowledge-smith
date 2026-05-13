---
arxiv: '2206.07682'
authors:
- Wei
- Tay
- Bommasani
- Raffel
- Zoph
- Borgeaud
- Yogatama
- Bosma
- Zhou
- Metzler
- Chi
- Hashimoto
- Vinyals
- Liang
- Dean
- Fedus (Google
- DeepMind
- Stanford)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2206.07682
  raw: '[[raw/papers/md/2022-emergent-abilities]]'
  source: https://arxiv.org/abs/2206.07682
owner: blaz
raw_pdf: raw/papers/pdf/2022-emergent-abilities.pdf
read: false
slug: emergent-abilities
tags:
- type/paper
- status/draft
- domain/evals
- domain/reasoning
- source/primary
title: Emergent Abilities of Large Language Models
type: note
updated: '2026-05-10'
year: 2022
---

# Emergent Abilities of Large Language Models

## Citation

- URL: https://arxiv.org/abs/2206.07682
- Authors: Wei, Tay, Bommasani, Raffel, Zoph, Borgeaud, Yogatama, Bosma, Zhou, Metzler, Chi, Hashimoto, Vinyals, Liang, Dean, Fedus (Google, DeepMind, Stanford)
- Year / venue: 2022 / TMLR 2022
- **Raw**: [[raw/papers/pdf/2022-emergent-abilities]]

## Core Claim

An ability is emergent if it is not present in smaller models but is present in larger models — performance is near-random until a critical scale threshold (~10²²–10²⁴ training FLOPs), then jumps to substantially above random. Emergent abilities cannot be predicted by extrapolating smaller-model performance.

## Key Paper Ideas

- **Definition of emergence**: near-random performance until critical scale, then sharp transition. Operationalized via scaling curves (FLOPs or parameters on x-axis, task performance on y-axis).
- **Two categories**: (a) few-shot prompting abilities that appear at scale, (b) augmented prompting strategies (CoT, instruction following) that are harmful/neutral at small scale, beneficial at large scale.
- **Cross-entropy vs downstream metrics**: Appendix A shows CE loss improves smoothly at scales where accuracy is flat/random. All 6 examined BIG-Bench tasks showed this pattern — downstream metrics can mask smooth underlying improvement.
- **Scale thresholds differ by model family**: PaLM 62B outperforms LaMDA 137B and GPT-3 175B on 14 BIG-Bench tasks, suggesting data quality and architecture matter.

## Methodology

Survey paper. Compiles scaling curves from GPT-3, LaMDA, Gopher, Chinchilla, PaLM across 26 abilities (13 few-shot, 13 augmented prompting). Manual classification of 210 BIG-Bench tasks as emergent, smoothly increasing, or flat.

## Key Results

- BIG-Bench arithmetic: near-zero until ~10²² FLOPs for GPT-3, ~10²³ for LaMDA
- MMLU: random guessing (25%) until ≥10²² FLOPs, substantially above random at 3-5×10²³
- CoT prompting: only surpasses standard prompting at ~10²³ FLOPs (~100B params)
- Cross-entropy improves smoothly even when accuracy is flat (all 6 examined tasks)

## Core Concepts

- [[concepts/evaluation-scaling-laws]] — emergence is the central phenomenon; metric choice determines observation
- [[concepts/benchmark-saturation]] — emergence is the opposite problem: benchmarks that are *too hard* at small scale
- [[maps/model-evaluation/landscape]] — implies eval suites must run across full range of scales

## Relevance To Poolside

Raises the practical question: if Poolside evaluates only at smaller scales during training, some capabilities may appear absent when they would emerge at the target scale. The Appendix A finding (CE loss improves smoothly) suggests perplexity tracking may be more informative than downstream accuracy at intermediate checkpoints.

## Blaz Notes

- 

## Related Notes

- Papers: [[notes/papers/2023-emergent-abilities-mirage]] (the rebuttal)
- Concepts: [[concepts/evaluation-scaling-laws]]
- Questions: [[questions/model-evaluation-methodology]] §proxy evals across scales

## Caveats

- Survey paper, not experimental — compiles existing curves
- Definition of emergence is based on visual inspection, no formal statistical test
- Sparse sampling of model scales (4-6 data points per family) makes it hard to distinguish sharp transitions from undersampled smooth curves
- The paper itself acknowledges the metric concern in Section 5.1 and Appendix A
