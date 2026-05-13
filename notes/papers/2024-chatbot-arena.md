---
arxiv: '2403.04132'
authors:
- Chiang
- Zheng
- Sheng
- Angelopoulos
- Li
- Li
- Zhu
- Zhang
- Jordan
- Gonzalez
- Stoica (UC Berkeley
- Stanford
- UCSD)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2403.04132
  raw: '[[raw/papers/md/2024-chatbot-arena]]'
  source: https://arxiv.org/abs/2403.04132
owner: blaz
raw_pdf: raw/papers/pdf/2024-chatbot-arena.pdf
read: false
slug: chatbot-arena
tags:
- type/paper
- status/draft
- domain/evals
- domain/general
- source/primary
title: 'Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference'
type: note
updated: '2026-05-10'
year: 2024
---

# Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

## Citation

- URL: https://arxiv.org/abs/2403.04132
- Authors: Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhu, Zhang, Jordan, Gonzalez, Stoica (UC Berkeley, Stanford, UCSD)
- Year / venue: 2024 / arXiv
- **Raw**: [[raw/papers/pdf/2024-chatbot-arena]]

## Core Claim

Chatbot Arena is a crowdsourced live evaluation platform using anonymous pairwise comparisons and Bradley-Terry modeling to produce statistically rigorous LLM rankings. 240K votes from 90K users across 50+ models. Crowdsourced votes show 72-83% agreement with expert evaluations (vs 79-90% expert-expert agreement).

## Key Paper Ideas

- **Live pairwise protocol**: anonymous models, user-generated prompts, identity revealed only after voting. Eliminates positional and identity bias.
- **Bradley-Terry over Elo**: BT coefficients via reweighted MLE targeting uniform distribution over model pairs. Better suited for statistical estimation with confidence intervals than running Elo.
- **Sandwich confidence intervals**: Huber (1967) robust standard errors for BT coefficients. Chi-squared multiplicity-corrected confidence set for approximate rankings.
- **Active (adaptive) sampling**: sample model pairs proportionally to expected CI width reduction. Requires 54% fewer samples than random sampling for win-matrix precision of 0.2.
- **Anomalous user detection**: exchangeability-based p-values combined via Fisher's test. ~90% TPR, 60-70% TNR.
- **Prompt diversity**: BERTopic pipeline (OpenAI embeddings → UMAP → HDBSCAN) identifies 600 clusters; largest is only 1% of total — genuine long-tail diversity.

## Methodology

Live platform with 240K votes, 90K users, 50+ models, 100+ languages (77% English). Four vote options: A wins, B wins, tie, both bad. Bradley-Terry model with sandwich CIs. Active sampling for efficient ranking.

## Key Results

| Finding | Value |
|---|---|
| Total votes | 240K |
| Users | 90K |
| Models | 50+ |
| Crowd-expert agreement | 72.8-83.1% |
| Expert-expert agreement | 79.4-89.8% |
| Adaptive vs random sampling | 54% fewer samples needed |
| Approximate votes for stable ranking | 10K-30K depending on model count |

GPT-4 win-rate vs Llama-2-70b varies from 96.7% (coding) to 53.3% (movie recommendations) — topic clusters discriminate at very different levels.

## Core Concepts

- [[concepts/llm-as-judge-methodology]] — Arena is the human-preference gold standard that LLM judges are validated against
- [[concepts/evaluation-variance]] — sandwich CIs quantify ranking uncertainty
- [[concepts/capability-decomposition]] — topic-specific win rates reveal capability structure

## Relevance To Poolside

Chatbot Arena rankings are the external gold standard against which Poolside models are compared. The Bradley-Terry methodology with adaptive sampling is directly relevant if Poolside ever builds internal preference-based evaluation. The topic-specific discriminability finding implies different eval domains provide different signal quality.

## Blaz Notes

- 

## Related Notes

- Papers: [[notes/papers/2023-judging-llm-as-a-judge]]
- Evals: [[evals/arena-hard-auto]]
- Concepts: [[concepts/llm-as-judge-methodology]]

## Caveats

- User base is LLM hobbyists/researchers, not representative of production usage
- 77% English — rankings may not generalize to non-English
- Focuses on helpfulness, not safety
- 10-20% crowd-expert disagreement, mainly from ambiguous prompts and missed factual errors
