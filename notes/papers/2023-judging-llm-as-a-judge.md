---
arxiv: '2306.05685'
authors:
- Zheng
- Chiang
- Sheng
- Wu
- Zhuang
- Lin
- Zhang
- Zhuang
- Li
- Li
- Xing
- Gonzalez
- Stoica (UC Berkeley
- UCSD
- CMU
- Stanford
- MBZUAI)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2306.05685
  raw: '[[raw/papers/md/2023-judging-llm-as-a-judge]]'
  source: https://arxiv.org/abs/2306.05685
owner: blaz
raw_pdf: raw/papers/pdf/2023-judging-llm-as-a-judge.pdf
read: false
slug: judging-llm-as-a-judge
tags:
- type/paper
- status/draft
- domain/evals
- domain/general
- source/primary
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
type: note
updated: '2026-05-10'
year: 2023
---

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

## Citation

- URL: https://arxiv.org/abs/2306.05685
- Authors: Zheng, Chiang, Sheng, Wu, Zhuang, Lin, Zhang, Zhuang, Li, Li, Xing, Gonzalez, Stoica (UC Berkeley, UCSD, CMU, Stanford, MBZUAI)
- Year / venue: 2023 / NeurIPS 2023 Datasets and Benchmarks
- **Raw**: [[raw/papers/pdf/2023-judging-llm-as-a-judge]]

## Core Claim

Strong LLM judges (GPT-4) match human agreement rates (>80%) on open-ended evaluation, making LLM-as-a-judge a scalable proxy for human preference. Introduces MT-Bench (80 multi-turn questions, GPT-4 single-answer grading) and Chatbot Arena (crowdsourced anonymous pairwise battles with Elo ranking).

## Key Paper Ideas

- **Judge mode taxonomy**: pairwise comparison (sensitive but quadratic cost), single-answer grading (scalable but less sensitive), reference-guided grading (reduces math failure from 70% to 15%)
- **Bias taxonomy**: position bias (GPT-4 consistent 65% under swap), verbosity bias (Claude-v1/GPT-3.5 fail 91% on repetitive-list attack, GPT-4 only 8.7%), self-enhancement bias (~10% win-rate inflation for GPT-4), limited reasoning capability
- **Position bias mitigation**: call judge twice with swapped order, require consistency; conservative (many ties)
- **Chain-of-thought judge**: solve first, then grade; reduces math grading failure from 70% to 30%
- **Agreement as validation metric**: probability that random individuals of two judge types agree on random question

## Methodology

**MT-Bench**: 80 hand-crafted multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, humanities knowledge). 10 per category, each with two turns. GPT-4 single-answer grading on 1-10 scale.

**Chatbot Arena**: crowdsourced anonymous pairwise battles. Users submit their own questions, interact with two anonymous models, vote. Model identities disclosed after voting. Bradley-Terry Elo rating from votes.

**Bias quantification**: systematic experiments swapping answer order, injecting repetitive content, comparing self-judge vs cross-judge win rates.

## Key Results

| Measurement | Value |
|---|---|
| GPT-4 vs human agreement (MT-bench, non-tied) | 85% |
| Human-human agreement (MT-bench, non-tied) | 81% |
| GPT-4 vs human agreement (Arena, non-tied) | 87% |
| Position bias consistency (GPT-4 / Claude-v1) | 65% / 23.8% |
| Verbosity attack failure (GPT-4 / GPT-3.5 / Claude-v1) | 8.7% / 91.3% / 91.3% |
| Math grading failure: default / CoT / reference | 70% / 30% / 15% |

When humans disagreed with GPT-4, they deemed GPT-4's judgment reasonable 75% of the time and changed their own choice 34% of the time.

## Core Concepts

- [[concepts/llm-as-judge-methodology]] — this paper is the canonical reference
- [[concepts/evaluation-variance]] — judge variance as a source of evaluation noise
- [[concepts/critic-validation]] — LLM-as-judge for evaluation is the model-eval instance of critic validation

## Relevance To Poolside

Defines the methodology underlying every judge-based eval in forge (Arena Hard, MT-Bench variants). The bias taxonomy (position, verbosity, self-enhancement) applies directly to any eval that uses LLM grading. The agreement-rate validation criterion (>80%) is the standard for trusting an LLM judge.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- How does judge quality degrade when using smaller/cheaper models as judges?
- Does the 80% agreement threshold transfer across languages and domains?
- Position bias mitigation via swapping is costly (2x calls) — are there cheaper alternatives?

## Related Notes

- Evals: [[evals/arena-hard-auto]], [[evals/mtbench]]
- Papers: [[notes/papers/2024-ai-agents-that-matter]]
- Concepts: [[concepts/llm-as-judge-methodology]], [[concepts/critic-validation]]

## Caveats

- Focuses on helpfulness only — neglects safety, honesty, harmlessness dimensions
- MT-bench is small (80 questions, 10 per category) — limited statistical power per category
- GPT-4 dependency: entire evaluation methodology depends on GPT-4's quality
- Self-enhancement bias analysis is inconclusive due to limited data
- Math/reasoning judging remains weak even with mitigations (15% failure with reference)
