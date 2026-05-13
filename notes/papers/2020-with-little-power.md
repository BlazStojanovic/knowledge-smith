---
arxiv: '2010.06595'
authors:
- Card
- Henderson
- Khandelwal
- Jia
- Mahowald
- Jurafsky (Stanford)
created: 2026-04-22
kind: paper
links:
  code: https://github.com/dallascard/NLP-power-analysis
  paper: https://arxiv.org/abs/2010.06595
  raw: '[[raw/papers/md/2020-with-little-power]]'
  source: https://arxiv.org/abs/2010.06595
owner: blaz
raw_pdf: raw/papers/pdf/2020-with-little-power.pdf
read: false
slug: with-little-power
tags:
- type/paper
- status/draft
- domain/evals
- domain/general
- source/primary
title: With Little Power Comes Great Responsibility
type: note
updated: '2026-05-10'
year: 2020
---

# With Little Power Comes Great Responsibility

## Citation

- URL: https://arxiv.org/abs/2010.06595
- Authors: Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky (Stanford)
- Year / venue: 2020 / EMNLP 2020
- **Raw**: [[raw/papers/pdf/2020-with-little-power]]

## Core Claim

Underpowered experiments are widespread in NLP. For several GLUE tasks (WNLI n=147, MRPC n=1725, SST-2 n=1821), test sets are too small to reliably detect typical improvements at current accuracy levels. At 98% accuracy, detecting a 5% relative improvement requires two orders of magnitude more data than at 80% accuracy.

## Key Paper Ideas

- **Minimum Detectable Effect (MDE)**: for each benchmark, the smallest improvement detectable with 80% power at α=0.05. WNLI MDE = 5.26% but mean reported improvement = 1.72% — hopelessly underpowered.
- **Simulation-based power analysis**: general algorithm applicable to any NLP eval. Define generative process G(n, e*, h), statistical test T, compute power as proportion of simulated datasets finding significance.
- **High-accuracy scaling**: at 98% accuracy with n=500, detecting δ=2% requires ~25% power (severely underpowered); at n=2000, power rises to ~80%.
- **MT power**: 2000-sentence test set has ~75% power for 1 BLEU point difference (below 80% threshold). Need ~5000 for well-powered 1-BLEU detection.
- **Human evaluation**: most common design (3 workers, 100 items) is underpowered for effects < 0.2 on [0,1] scale.
- **Retropower**: only 46% of surveyed SOTA claims on GLUE would have had ≥80% predicted power.

## Methodology

Power analysis via simulation for three settings: accuracy comparisons (McNemar's test), BLEU comparisons (randomization test with Delta-Laplace mixture), human evaluations (mixed-effects models). Regression on GLUE leaderboard data to predict expected effect sizes and agreement rates.

## Key Results

| Benchmark | Test Size | SOTA | MDE | Mean Reported Δ |
|---|---|---|---|---|
| WNLI | 147 | 94.5% | +5.26% | +1.72% |
| MRPC | 1,725 | 92.0% | +1.62% | +0.63% |
| SST-2 | 1,821 | 97.2% | +1.02% | +0.57% |
| QQP | 390,965 | 91.0% | +0.11% | +0.36% |

- MT with 2000 sentences: ~75% power for 1 BLEU
- Human evals (3 workers × 100 items): underpowered for effects < 0.2
- Type-M error at n=500, δ=2%, Pa=0.9: significant findings exaggerate by 1.9×

## Core Concepts

- [[concepts/evaluation-variance]] — power analysis quantifies when variance overwhelms signal
- [[concepts/benchmark-saturation]] — underpoweredness at high accuracy is a form of saturation
- [[maps/model-evaluation/statistical-reliability]] — power analysis is a key reliability diagnostic

## Relevance To Poolside

Directly applicable to Poolside's smaller benchmarks: GPQA Diamond (198 questions), IFBench (294 samples), DROP, etc. The MDE framework provides a concrete check: given the test set size and expected frontier accuracy, can this eval detect meaningful improvements? Benchmarks below the MDE threshold provide noise, not signal, for training decisions.

## Blaz Notes

- 

## Related Notes

- Papers: [[notes/papers/2018-statistical-significance-in-nlp]], [[notes/papers/2021-what-will-it-take-to-fix-benchmarking]]
- Concepts: [[concepts/evaluation-variance]], [[concepts/benchmark-saturation]]

## Caveats

- Regression-based effect prediction has R²=0.69 — imperfect prior
- MT analysis fitted on single language pair (En-De)
- Pre-LLM era — doesn't address pass@k variance or LLM-judge variance
- Acknowledges NHST framework limitations but argues it still provides value
