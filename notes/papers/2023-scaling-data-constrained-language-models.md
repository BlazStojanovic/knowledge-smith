---
arxiv: '2305.16264'
authors:
- Niklas Muennighoff
- Alexander M. Rush
- Boaz Barak
- Teven Le Scao
- Aleksandra Piktus
- Nouamane Tazi
- Sampo Pyysalo
- Thomas Wolf
- Colin Raffel
created: 2026-04-22
kind: paper
links:
  code: https://github.com/huggingface/datablations
  paper: https://jmlr.org/papers/v26/24-1000.html
  raw: '[[raw/papers/md/2023-scaling-data-constrained-language-models.notes]]'
  source: https://jmlr.org/papers/v26/24-1000.html
owner: blaz
raw_pdf: raw/papers/pdf/2023-scaling-data-constrained-language-models.pdf
read: false
slug: scaling-data-constrained-language-models
tags:
- type/paper
- status/verified
- domain/llm
- domain/pretraining
- domain/repetition
- domain/synth-data
- source/paper
- confidential/public-source
title: Scaling Data-Constrained Language Models
type: note
updated: '2026-05-10'
year: 2023
---

# Scaling Data-Constrained Language Models

## Citation

- URL: https://jmlr.org/papers/v26/24-1000.html
- PDF: https://jmlr.org/papers/volume26/24-1000/24-1000.pdf
- arXiv: https://arxiv.org/abs/2305.16264
- Authors: Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel
- Venue: JMLR 26(53), 2025; originally NeurIPS 2023 oral
- Raw: [[raw/papers/pdf/2023-scaling-data-constrained-language-models.notes]]

## Short Summary

Studies language-model scaling when unique text data is constrained. The key prior for Poolside is that full-dataset repetition up to roughly 4 epochs can be close to fresh-token training in the paper's setting, while larger repetition eventually has diminishing value.

## Core Claim

The paper studies language-model scaling when unique text data is constrained. Its relevant finding for DAT-440 is that, at fixed compute, training with repeated data for up to 4 epochs causes negligible loss change compared with using unique data; beyond that, extra repetition has diminishing value.

## Method

- Trains over 400 models across data and compute constraints.
- Varies data repetition, model size, and training tokens.
- Fits a data-constrained scaling law that treats repeated tokens as having decreasing value.
- Evaluates loss and downstream task behavior in repeated-data regimes.

## Key Results

- Repeating the full dataset up to 4 epochs is close to unique-data training for loss in the studied setting.
- More repetition eventually gives diminishing returns, and the value of additional compute can decay toward zero.
- In data-constrained regimes, compute-optimal allocation differs from naive Chinchilla-style assumptions because repeated tokens are not equivalent to fresh tokens.

## Relevance To Poolside

Our interpretation: this is the source for the "up to roughly 4x may be okay" prior in [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]. It does not prove that repeated seed data is harmless for Laguna XS code-data runs; it gives a broader LLM pretraining prior that DAT-440 should test in Poolside's setting.

For DAT-440, the important distinction is:

- Paper claim: repeated full-dataset epochs can be acceptable up to about 4 epochs under the paper’s conditions.
- Poolside hypothesis: localized repetition of seed code data at Laguna XS may still hurt downstream model quality, and raw-code rephrasing may mitigate it.

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Repetition mode: full-dataset epochs under data-constrained pretraining.
- Memo-grounded claim: up to about 4 epochs is nearly loss-neutral in the studied setting; meaningful gains can persist beyond that, but repeated tokens eventually saturate in value.
- Memo-grounded contrast: this result is more benign than Hernandez et al.'s partial-repetition setting and should not be applied blindly to up-weighted seed subsets.

## Related Notes

- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
- [[experiments/dat-440-raw-code-rephrasing]]
- [[technical/raw-code-rephrasing-pipeline]]
- [[concepts/scaling-laws-foundational]] — Muennighoff's effective-token model extends Chinchilla
- [[concepts/data-repetition]] — full-dataset epoch regime
- [[maps/scaling-laws/landscape]] — data-constrained domain

## Caveats

- The paper studies broad data-constrained pretraining regimes, not this exact Poolside data mix.
- The 4-epoch prior is about repeated data under fixed-compute scaling experiments; it should not be treated as a universal safe threshold.
- The paper’s claim is primarily loss-focused; DAT-440 should judge downstream Laguna XS model metrics.
