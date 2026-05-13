---
arxiv: '2308.06259'
authors:
- Xian Li
- Ping Yu
- Chunting Zhou
- Timo Schick
- Omer Levy
- Luke Zettlemoyer
- Jason Weston
- Mike Lewis (Meta AI)
created: 2026-04-29
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2308.06259
  raw: https://arxiv.org/pdf/2308.06259
  source: https://arxiv.org/abs/2308.06259
owner: blaz
read: false
slug: self-alignment-with-instruction-backtranslation
tags:
- type/paper
- status/stub
- source/primary
- domain/synth-data
- domain/sft
- domain/llm
title: Self-Alignment with Instruction Backtranslation
type: note
updated: '2026-05-10'
year: 2023
---

# Self-Alignment with Instruction Backtranslation

## Citation

- URL: https://arxiv.org/abs/2308.06259
- PDF: https://arxiv.org/pdf/2308.06259
- Authors: Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis (Meta AI)
- Year / venue: arXiv Aug 2023; ICLR 2024

## Core Claim

A self-alignment recipe that builds an instruction-following LLM by (i) using a seed-finetuned model to generate candidate instructions for unlabelled web documents (*self-augmentation*), (ii) selecting high-quality (instruction, response) pairs via the same model (*self-curation*), and (iii) finetuning a stronger model on the resulting data. Two iterations on LLaMa beat all non-distilled LLaMa models on the Alpaca leaderboard.

## Key Paper Ideas

- **Instruction backtranslation**: web document treated as the *response*; the model generates the corresponding *instruction*. Inverts the usual instruction → response direction and lets web text be reused as alignment data without human annotation or teacher distillation.
- **Self-curation**: same model scores its own (instruction, response) candidates and keeps top-quality pairs.
- **No distillation**: contrasts with Alpaca-/Vicuna-style pipelines that depend on a stronger teacher.

## Methodology

- Seed: small human-labelled instruction-following set; finetune a base LLaMa on it.
- Self-augment: for each web document $d$, prompt the seed model to produce instruction $i$ such that $d$ is a valid response → candidate pair $(i, d)$.
- Self-curate: same model rates candidate pairs; keep high-quality.
- Finetune a fresh base model on (seed ∪ curated). Repeat for 2 iterations.

## Experiments

- Two iterations on LLaMa; evaluate on Alpaca leaderboard.

## Key Results

- Outperforms all other LLaMa-based models on the Alpaca leaderboard that do *not* use distillation data.

## Core Concepts

- Existing concepts: [[concepts/self-alignment-vs-distillation]], [[concepts/rephrasal-operations]] (archetype #5 cross-modal-ish: response↔instruction), [[concepts/synthetic-data-formalism]], [[concepts/content-provenance-axis]] (the *response* side is web-grounded → Tier 2 seed-dominant; the *instruction* side is generator-bound — interesting hybrid).
- Concepts to extract: —

## Relevance To Poolside

*Stub.* The instruction-backtranslation primitive (use natural artefact as response, synthesise instruction) is the cleanest existing template for "code as response, instruction as synthesis" alignment data — directly analogous to Magicoder OSS-Instruct and StarCoder2-Instruct's "snippet → synthesised instruction → response" pipeline.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- 

## Related Notes

- Concepts: [[concepts/self-alignment-vs-distillation]], [[concepts/rephrasal-operations]], [[concepts/content-provenance-axis]]

## Caveats

- Stub note; abstract-level only.
