---
arxiv: '2309.11495'
authors:
- Shehzaad Dhuliawala
- Mojtaba Komeili
- Jing Xu
- Roberta Raileanu
- Xian Li
- Asli Celikyilmaz
- Jason Weston
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2309.11495
  raw: null
  source: https://arxiv.org/abs/2309.11495
owner: blaz
read: false
slug: chain-of-verification-reduces-hallucination
tags:
- type/paper
- source/primary
- status/stub
- domain/reasoning
- domain/llm
title: Chain-of-Verification Reduces Hallucination in Large Language Models
type: note
updated: '2026-05-10'
year: 2024
---

# Chain-of-Verification Reduces Hallucination in Large Language Models

Prompts the model to cross-check its own outputs via planned verification questions (CoVe).

- **Authors**: Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston
- **Venue**: ACL 2024
- **arXiv**: [2309.11495](https://arxiv.org/abs/2309.11495)
- **Raw**: [[raw/papers/pdf/2024-chain-of-verification]]

## Core contribution

CoVe (Chain-of-Verification): after generating an initial response, the model plans verification questions, answers them independently to avoid bias, and then produces a revised response. Reduces hallucination across list-based, closed-book QA, and long-form generation tasks.

## Connections

- Related: [[notes/papers/2023-rarr-researching-and-revising-what-language-models-say]]
- Related: [[notes/papers/2023-selfcheckgpt]]
