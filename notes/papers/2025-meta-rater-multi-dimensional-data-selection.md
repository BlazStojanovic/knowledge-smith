---
aliases:
- Meta-rater
- meta-rater
arxiv: '2504.14194'
authors:
- Xinlin Zhuang
- Jiahui Peng
- Ren Ma
- Yinfan Wang
- Tianyi Bai
- Xingjian Wei
- Jiantao Qiu
- Chi Zhang
- Ying Qian
- Conghui He (Shanghai AI Lab
- East China Normal University)
created: 2026-04-23
kind: paper
links:
  code: https://github.com/opendatalab/Meta-rater
  paper: https://arxiv.org/abs/2504.14194
  raw: '[[raw/papers/md/2025-meta-rater-multi-dimensional-data-selection]]'
  source: https://arxiv.org/abs/2504.14194
owner: blaz
raw_pdf: raw/papers/pdf/2025-meta-rater-multi-dimensional-data-selection.pdf
read: false
slug: meta-rater-multi-dimensional-data-selection
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language
  Models'
type: note
updated: '2026-05-10'
year: 2025
---

# Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models

## Citation

- URL: https://arxiv.org/abs/2504.14194
- PDF: https://arxiv.org/pdf/2504.14194
- Authors: Xinlin Zhuang, Jiahui Peng, Ren Ma, Yinfan Wang, Tianyi Bai, Xingjian Wei, Jiantao Qiu, Chi Zhang, Ying Qian, Conghui He (Shanghai AI Lab, East China Normal University)
- Year / venue: 2025-04 arXiv preprint
- Raw PDF: [[raw/papers/pdf/2025-meta-rater-multi-dimensional-data-selection.pdf]]

## Core Claim

Integrating multiple quality dimensions through learned optimal weightings via proxy-model regression (Meta-rater) doubles convergence speed and improves downstream task performance by 3.23% for 1.3B models, outperforming single-dimensional data selection approaches.

## Key Paper Ideas

- **PRRC Framework**: four novel evaluation dimensions — Professionalism, Readability, Reasoning, Cleanliness — designed to complement existing quality signals.
- **Learned optimal weighting**: rather than manually tuning quality-score weights, Meta-rater trains hundreds of small proxy models with random weight combinations, fits a LightGBM regression model predicting validation loss from weight vectors, then searches for the optimal weight combination.
- **25 quality metrics combined**: integrates NL quality signals (RedPajama heuristics), data importance scores (DSIR with Book/Wikipedia/AutoMathText references), model-based ratings (QuRating 4 dims, FineWeb-Edu educational value, WanjuanCC advertisement/fluency), and the 4 new PRRC dimensions.
- **Rating models**: fine-tuned ModernBERT classifiers for each PRRC dimension, achieving 87–92% F1.
- **Annotated SlimPajama-627B**: full corpus annotated with all 25 quality metrics (publicly released).

## Methodology

1. Annotate corpus with $m$ quality scores per document.
2. Sample $N$ random weight vectors $\mathbf{w}_i$ for quality-score aggregation.
3. For each $\mathbf{w}_i$: select top-$k$ data by weighted score $Q_{agg}(x) = \sum_j w_{ij} \cdot Q_j(x)$, train a small proxy model, record validation loss $l_i$.
4. Fit regression model $f(\mathbf{w}) \to \hat{l}$ predicting loss from weights.
5. Search for optimal weights $\mathbf{w}^* = \arg\min_{\tilde{\mathbf{w}}} f(\tilde{\mathbf{w}})$.
6. Select final training data using $\mathbf{w}^*$.

## Key Results

- **2× convergence speed**: 1.3B model trained on Meta-rater-selected data matches random-selection performance at half the training tokens.
- **+3.23% downstream accuracy**: averaged across general knowledge, commonsense reasoning, reading comprehension benchmarks.
- **Scales**: validated on 3.3B (100B tokens) and 7.2B models — performance gains persist.
- **Optimal weights**: analysis reveals which quality dimensions contribute most varies by downstream task category.

## Core Concepts

- Existing concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]]

## Relevance To Poolside

*Our interpretation.* Meta-rater demonstrates that multi-dimensional quality integration outperforms any single quality score, even when the individual dimensions include state-of-the-art signals like QuRating and FineWeb-Edu. The proxy-model-based weight optimization is computationally practical (hundreds of small model runs, not a hyperparameter grid search on the full model). For Poolside, this approach could optimize weights across code-specific quality dimensions (test coverage, reasoning depth, complexity) alongside general quality signals for seed data selection.

## Key Follow-Ups / Jumping-Off Points

- Direct comparison with [[notes/papers/2024-qurating-selecting-high-quality-data-for-training-language-models]] (single best dimension: QuRating educational value).
- Uses [[notes/papers/2025-dataman-data-manager-for-pretraining-large-language-models]] criteria as input.
- Connection to [[notes/papers/2024-scaling-laws-for-data-filtering]] — optimal quality threshold depends on compute budget; Meta-rater's weight optimization could interact with scaling-law-aware threshold selection.

## Related Notes

- Concepts: [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]]
- Maps: [[maps/evaluation/point-level]]

## Caveats

- Proxy models (small scale, short training) may not capture quality-dimension interactions that matter at large scale.
- Weight optimization is compute-intensive: requires hundreds of proxy model training runs.
- Only validated on English web text (SlimPajama).
