---
arxiv: '2310.17623'
authors:
- Yonatan Oren
- Nicole Meister
- Niladri Chatterji
- Faisal Ladhak
- Tatsunori B. Hashimoto
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2310.17623
  raw: null
  source: https://arxiv.org/abs/2310.17623
owner: blaz
raw_pdf: raw/papers/pdf/2023-proving-test-set-contamination.pdf
read: false
slug: proving-test-set-contamination
tags:
- type/paper
- source/primary
- status/stub
- domain/evals
title: Proving Test Set Contamination in Black Box Language Models
type: note
updated: '2026-05-10'
year: 2023
---

# Proving Test Set Contamination in Black Box Language Models

Provable guarantees of test set contamination without access to pretraining data or model weights, exploiting memorized example ordering.

- **Authors**: Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto
- **Venue**: arXiv 2023
- **arXiv**: [2310.17623](https://arxiv.org/abs/2310.17623)
- **Raw**: [[raw/papers/pdf/2023-proving-test-set-contamination]]

## Core contribution

Exploits how models memorize example ordering: compares likelihood of canonically ordered datasets against shuffled versions. If the model assigns higher likelihood to the canonical order, this constitutes evidence of contamination. The method is black-box — requires only generation/likelihood access, not weights.

## Key results

- Procedure sensitive enough to detect contamination in models as small as 1.4B parameters
- Little evidence for pervasive contamination across five audited public models
- Provides statistical guarantees (exchangeability test)

## Connections

- Map: [[maps/model-evaluation/contamination-methods]] — black-box, behavioral signal (ordering)
- Concept: [[concepts/benchmark-contamination]]
