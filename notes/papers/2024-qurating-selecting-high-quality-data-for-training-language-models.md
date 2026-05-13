---
aliases:
- QuRating
- qurating
arxiv: '2402.09739'
authors:
- Alexander Wettig
- Aatmik Gupta
- Saumya Malik
- Danqi Chen (Princeton)
created: 2026-04-23
kind: paper
links:
  code: https://github.com/princeton-nlp/QuRating
  paper: https://arxiv.org/abs/2402.09739
  raw: '[[raw/papers/md/2024-qurating-selecting-high-quality-data-for-training-language-models]]'
  source: https://arxiv.org/abs/2402.09739
owner: blaz
raw_pdf: raw/papers/pdf/2024-qurating-selecting-high-quality-data-for-training-language-models.pdf
read: false
slug: qurating-selecting-high-quality-data-for-training-language-models
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'QuRating: Selecting High-Quality Data for Training Language Models'
type: note
updated: '2026-05-10'
year: 2024
---

# QuRating: Selecting High-Quality Data for Training Language Models

## Citation

- URL: https://arxiv.org/abs/2402.09739
- PDF: https://arxiv.org/pdf/2402.09739
- Authors: Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen (Princeton)
- Year / venue: 2024, ICML 2024 (PMLR 235)
- Raw PDF: [[raw/papers/pdf/2024-qurating-selecting-high-quality-data-for-training-language-models.pdf]]

## Core Claim

Pairwise LLM judgments along four quality criteria (writing style, facts & trivia, educational value, required expertise) can be distilled into a small QuRater model that assigns scalar quality ratings to a 260B-token corpus, enabling quality-based data selection that outperforms uniform sampling and perplexity filtering.

## Key Paper Ideas

- **Four quality criteria**: writing style, facts & trivia, educational value, required expertise. Chosen to be (1) applicable to varied text, (2) requiring deep content understanding, (3) complementary to each other.
- **Pairwise > pointwise**: LLMs produce more stable and discriminative judgments in pairwise comparisons than absolute ratings (Kendall tau 0.79 pairwise vs. 0.61 pointwise).
- **Bradley-Terry model**: pairwise preferences $p_{B \succ A} = \sigma(s_B - s_A)$ translated into per-document scalar ratings via maximum-likelihood estimation.
- **Temperature-controlled selection**: sampling probability $p(d_i) \propto \exp(s_i / \tau)$ where $\tau$ trades off quality ($\tau \to 0$, top-$k$) vs. diversity ($\tau \to \infty$, uniform). Best results at $\tau = 2.0$.
- **Quality ratings as rewards**: the sampling scheme implicitly changes the language modelling objective toward reward-weighted regression (connection to RLHF).
- **Curriculum from quality**: training on examples ordered by quality rating (low-to-high) outperforms random order, without changing the dataset.

## Methodology

- GPT-3.5-turbo judges 250K text pairs per criterion (both orderings to counteract positional bias).
- Fine-tune 1.3B Sheared-LLaMA (Xia et al. 2024) on pairwise judgments with four linear heads (one per criterion) — QuRater model. >93% accuracy on held-out judgments.
- Annotate 260B tokens from SlimPajama → QuRatedPajama (publicly released).
- Select 30B tokens using temperature-controlled sampling per criterion; train 1.3B models from scratch.

## Key Results

- **Educational value** at $\tau = 2.0$ improves ICL by 1.8% avg over uniform, matching a model trained with 50% more tokens.
- **Writing style** yields best perplexity but does not lead to substantial downstream improvements.
- **Facts & trivia** selection helps knowledge-intensive tasks (NQ, MMLU) but hurts reading comprehension.
- Top-$k$ selection ($\tau = 0$) excels at specific tasks but underperforms across the board — balancing quality and diversity matters.
- Quality-based curriculum (ordered low→high) outperforms random-order training on identical data.

## Core Concepts

- Existing concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]]
- Concepts to extract: quality-diversity tradeoff in data selection (temperature parameter)

## Relevance To Poolside

*Our interpretation.* QuRating demonstrates that multi-dimensional quality ratings enable more compositional data selection than single-score approaches. The temperature-based quality-diversity tradeoff is directly applicable to seed data selection for synthetic pipelines — select high-quality seeds while maintaining diversity. The curriculum finding (ordering by quality) could apply to synthetic data scheduling.

## Key Follow-Ups / Jumping-Off Points

- [[notes/papers/2025-dataman-data-manager-for-pretraining-large-language-models]] — extends to 14 criteria using pointwise rating (argued more practical than pairwise at scale).
- [[notes/papers/2025-meta-rater-multi-dimensional-data-selection]] — integrates QuRating's 4 dimensions with other signals via learned weighting.

## Related Notes

- Concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]], [[concepts/verification-signals]]
- Maps: [[maps/evaluation/point-level]]

## Caveats

- Pairwise annotation is expensive: 520 H100-hours for 260B tokens.
- Only tested at 1.3B model scale on 30B tokens.
- Quality criteria chosen by human intuition; no systematic search over criteria space.
