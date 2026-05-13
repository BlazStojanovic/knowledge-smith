---
arxiv: '2403.08540'
authors:
- Gadre
- Smyrnis
- Shankar
- Gururangan
- Wortsman
- Shao
- Mercat
- Fang
- Li
- Keh
- Xin
- Nezhurina
- Vasiljevic
- Jitsev
- Soldaini
- Dimakis
- Ilharco
- Koh
- Song
- Kollar
- Carmon
- Dave
- Heckel
- Muennighoff
- Schmidt (Columbia
- TRI
- UT Austin
- Apple
- UW
- AI2
- Stanford
- TAU
- TU Munich
- Contextual AI)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2403.08540
  raw: '[[raw/papers/md/2024-over-training-scaling]]'
  source: https://arxiv.org/abs/2403.08540
owner: blaz
raw_pdf: raw/papers/pdf/2024-over-training-scaling.pdf
read: false
slug: over-training-scaling
tags:
- type/paper
- status/draft
- domain/evals
- domain/training
- source/primary
title: Language Models Scale Reliably with Over-Training and on Downstream Tasks
type: note
updated: '2026-05-10'
year: 2024
---

# Language Models Scale Reliably with Over-Training and on Downstream Tasks

## Citation

- URL: https://arxiv.org/abs/2403.08540
- Authors: Gadre, Smyrnis, Shankar, Gururangan, Wortsman, Shao, Mercat, Fang, Li, Keh, Xin, Nezhurina, Vasiljevic, Jitsev, Soldaini, Dimakis, Ilharco, Koh, Song, Kollar, Carmon, Dave, Heckel, Muennighoff, Schmidt (Columbia, TRI, UT Austin, Apple, UW, AI2, Stanford, TAU, TU Munich, Contextual AI)
- Year / venue: 2024 / arXiv
- **Raw**: [[raw/papers/pdf/2024-over-training-scaling]]

## Core Claim

Scaling laws extrapolate reliably in both the over-trained regime (training beyond compute-optimal) and for predicting downstream task performance. A 1.4B model at 32× over-training predicted from 300× less compute (0.7% relative error on loss); average downstream error for a 6.9B model predicted from 20× less compute (0.05% relative error on 17-task average).

## Key Paper Ideas

- **Reparameterized scaling law for over-training**: L(C, M) = E + (aM^η + bM^{-η})·C^{-η} where M = D/N (token multiplier). Power-law exponent η is invariant to over-training level — log-log plots show parallel lines.
- **Power law for downstream error**: Err(L) = ε - k·exp(-γ·L). Average top-1 error follows exponential decay as loss decreases. Chain: (C, M) → L → Err.
- **Three-step prediction pipeline**: (1) fit loss scaling from small runs, (2) fit error-vs-loss from moderate runs, (3) chain to predict downstream error at target compute.
- **Aggregate over individual**: average error across 17 tasks is predictable (0.05% relative error) but individual task error is noisy (up to 80% relative error). Argues for aggregate evaluation.
- **1.4B anchor requirement**: downstream prediction critically depends on a moderately-sized calibration run. Removing 1.4B anchor from fit causes error to jump from 0.05% to 10.64%.

## Methodology

104 models, 0.011B-6.9B params, trained on C4/RedPajama/RefinedWeb. Token multipliers M=5-640 (M=20 ≈ compute-optimal). 435-model hyperparameter sweep filtered to Pareto frontier. Levenberg-Marquardt fitting. 17-task downstream eval filtered to tasks where ≥1 small model achieves 10pp above chance.

## Key Results

| Prediction | Compute Savings | Relative Error |
|---|---|---|
| 1.4B loss (32× over-trained) | 300× | 0.7% |
| 6.9B loss (compute-optimal) | 300× | 0.7% |
| 6.9B 17-task avg error | 20× | 0.05% |
| 1.4B 17-task avg error (32× OT) | 20× | 3.6% |

Fitting cost: ~100 A100-hrs for loss scaling, ~1000 A100-hrs for downstream error scaling.

## Core Concepts

- [[concepts/evaluation-scaling-laws]] — this paper provides the most actionable prediction methodology
- [[maps/model-evaluation/landscape]] — eval scheduling during training
- [[concepts/capability-decomposition]] — aggregate is predictable, individual tasks are not

## Relevance To Poolside

Directly applicable to Poolside's training pipeline: before committing to a large training run, fit scaling laws from smaller runs to predict both loss and downstream performance. The 1.4B anchor requirement means Poolside should invest in a moderate calibration run. The aggregate-over-individual finding argues for using PRETRAINING_DEFAULT suite averages rather than individual benchmark scores for training decisions. The out-of-distribution failure (English training → code eval) is a caution for cross-domain extrapolation.

## Blaz Notes

- 

## Related Notes

- Papers: [[notes/papers/2022-training-compute-optimal-large-language-models]], [[notes/papers/2023-emergent-abilities-mirage]]
- Concepts: [[concepts/evaluation-scaling-laws]]
- Questions: [[questions/model-evaluation-methodology]] §eval cadence, §proxy evals

## Caveats

- Validated only up to 6.9B — trends at 70B+ unverified
- No post-training: does not consider SFT/RLHF effects on scaling
- Individual task prediction is noisy (up to 80% relative error)
- Scaling law parameters must be re-fit per training distribution
- Requires expensive hyperparameter sweep (435 models) and moderate anchor model
