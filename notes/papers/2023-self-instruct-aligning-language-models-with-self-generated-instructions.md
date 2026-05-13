---
arxiv: '2212.10560'
authors:
- Yizhong Wang
- Yeganeh Kordi
- Swaroop Mishra
- Alisa Liu
- Noah A. Smith
- Daniel Khashabi
- Hannaneh Hajishirzi
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2212.10560
  raw: null
  source: https://arxiv.org/abs/2212.10560
owner: blaz
read: false
slug: self-instruct-aligning-language-models-with-self-generated-instructions
tags:
- type/paper
- source/primary
- status/stub
- domain/synth-data
- stage/sft
title: 'Self-Instruct: Aligning Language Models with Self-Generated Instructions'
type: note
updated: '2026-05-10'
year: 2023
---

# Self-Instruct: Aligning Language Models with Self-Generated Instructions

Bootstrapping instruction-following data from a model's own generations using seed tasks.

- **Authors**: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Venue**: ACL 2023
- **arXiv**: [2212.10560](https://arxiv.org/abs/2212.10560)
- **Raw**: [[raw/papers/pdf/2023-self-instruct]]

## Core contribution

Establishes the paradigm of extracting a model's internal knowledge into labeled instruction-following datasets by prompting it to generate diverse instances from a small set of seed tasks. Shows that a vanilla GPT-3 fine-tuned on its own self-generated instructions approaches InstructGPT performance.

## Connections

- Successor: [[notes/papers/2024-evol-instruct-wizardlm]]
