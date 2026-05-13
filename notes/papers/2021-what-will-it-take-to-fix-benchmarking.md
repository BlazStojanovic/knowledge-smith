---
arxiv: '2104.02145'
authors:
- Samuel R. Bowman (NYU)
- George E. Dahl (Google Brain)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2104.02145
  raw: '[[raw/papers/md/2021-what-will-it-take-to-fix-benchmarking]]'
  source: https://arxiv.org/abs/2104.02145
owner: blaz
raw_pdf: raw/papers/pdf/2021-what-will-it-take-to-fix-benchmarking.pdf
read: false
slug: what-will-it-take-to-fix-benchmarking
tags:
- type/paper
- status/draft
- domain/evals
- domain/general
- source/primary
title: What Will it Take to Fix Benchmarking in Natural Language Understanding?
type: note
updated: '2026-05-10'
year: 2021
---

# What Will it Take to Fix Benchmarking in Natural Language Understanding?

## Citation

- URL: https://arxiv.org/abs/2104.02145
- Authors: Samuel R. Bowman (NYU), George E. Dahl (Google Brain)
- Year / venue: 2021 / arXiv position paper
- **Raw**: [[raw/papers/pdf/2021-what-will-it-take-to-fix-benchmarking]]

## Core Claim

NLU benchmarks are broken: unreliable and biased systems score so highly that researchers with genuinely better systems cannot demonstrate improvements. Adversarial out-of-distribution test sets are not the fix — they obscure the abilities we want to measure. Four criteria define a sound benchmark: validity, annotation reliability, statistical power, and disincentives for bias.

## Key Paper Ideas

- **Four benchmark quality criteria**: validity (benchmark reflects the task), annotation reliability (labels are accurate/unambiguous), statistical power (large/hard enough to detect real improvements), bias disincentives (reveals and discourages harmful biases)
- **Adversarial filtering critique**: adversarial construction is neither necessary nor sufficient — it is "mode-seeking" rather than "mass-covering," incentivizing models to be *different* rather than *better*
- **Statistical power at high accuracy**: at 98% accuracy, detecting a 5% relative improvement requires two orders of magnitude more data than at 80% accuracy (Card et al. 2020)
- **Annotation ambiguity as false superhuman**: ~20% of textual entailment examples are significantly ambiguous (Pavlick & Kwiatkowski 2019); models outperform individual humans by learning the most-frequent label, not the task
- **Data construction taxonomy**: naturally-occurring (poor validity for multi-input), expert-authored (biased coverage), crowdsourced (repetitive/easy), adversarially-filtered (mode-seeking)

## Methodology

Position paper analyzing four benchmark quality criteria across existing NLU benchmarks. No experiments — structured argument with citations.

**Proposed solutions**:
- Validity: hybrid data collection (crowdsourcing + expert augmentation, linguist-guided interventions)
- Annotation: multiple redundant annotations, validation phases, predicting label distributions
- Statistical power: invest in larger datasets (10-way annotated 500K examples costs >$1M but is cheaper than wasted researcher time)
- Bias: auxiliary bias metrics as families of expert-constructed test sets, community infrastructure to enforce reporting

## Core Concepts

- [[concepts/benchmark-saturation]] — this paper provides the foundational argument (validity + statistical power failure)
- [[concepts/evaluation-variance]] — annotation ambiguity as a variance source
- [[concepts/temporal-benchmark-robustness]] — adversarial construction as a flawed robustness strategy

## Relevance To Poolside

The four-criteria framework is a checklist for evaluating any Poolside benchmark's methodological soundness. The statistical power argument explains why small benchmarks (GPQA Diamond: 198 questions) are risky at frontier accuracy. The adversarial filtering critique is relevant to dynamic benchmark strategies.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- Can the validity criterion be operationalized into a measurable diagnostic for existing benchmarks?
- The $1M cost estimate for a robust 500K-example benchmark — has this become cheaper with LLM-assisted annotation?
- What does the adversarial filtering critique imply for GSM-Infinite / Dynabench-style dynamic benchmarks?

## Related Notes

- Papers: [[notes/papers/2022-helm]], [[notes/papers/2023-judging-llm-as-a-judge]]
- Concepts: [[concepts/benchmark-saturation]], [[concepts/evaluation-variance]]
- Maps: [[maps/model-evaluation/landscape]]

## Caveats

- Position paper with no experiments — arguments are conceptual, not empirically validated
- Pre-LLM era (2021) — does not address LLM-specific evaluation challenges
- The hybrid data collection proposal is aspirational — no practical implementation demonstrated
- 6 pages — necessarily high-level on each criterion
