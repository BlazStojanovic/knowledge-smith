---
arxiv: '2407.21077'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.21077
  raw: null
  source: https://arxiv.org/abs/2407.21077
owner: blaz
raw_pdf: raw/papers/pdf/2024-genetic-instruct.pdf
read: false
slug: genetic-instruct
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- stage/sft
title: Genetic Instruct
type: note
updated: '2026-05-10'
year: 2024
---

# Genetic Instruct

- **arXiv**: [2407.21077](https://arxiv.org/abs/2407.21077)
- **Authors / affiliation**: Somshubra Majumdar, Vahid Noroozi, Sean Narenthiran, Aleksander Ficek, Jagadeesh Balam, Boris Ginsburg (NVIDIA)
- **Year**: 2024
- **Raw**: [[raw/papers/pdf/2024-genetic-instruct]]
- **Grounding axis**: [[maps/grounding/ungrounded-seed]] (seed-pool evolution)
- **Output shape**: (instruction, code) pair
- **Filter / verification**: Judge-LLM fitness scoring. No execution.
- **Training stage**: SFT

## Method

Evolutionary loop over an instruction pool seeded with a small set of initial code-instruction samples. Three roles played by LLMs:

- **Instructor-LLM** — applies **crossover** (few-shot recombination of parent instructions, adapted from Self-Instruct) and **mutation** (rewrite an instruction under predefined rules) to generate new instructions.
- **Coder-LLM** — synthesises a solution given a generated instruction.
- **Judge-LLM** — scores candidate (instruction, code) pairs for automatic quality evaluation / fitness.

Generated instructions augment the pool; filtered pairs enter the SFT corpus.

## Key result

Reported scaling to >7.5M coding instructions; the paper reports improvements in code-generation capability vs. prior synthetic-generation approaches and publicly available datasets when SFT-ing on the Genetic-Instruct corpus. Verify specific pass@1 numbers against the paper's main table before quoting.

## Notes

- Filter is purely LLM-as-judge; no execution grounding.
- Cross-cuts [[concepts/verification-signals]] §LLM-as-judge and [[maps/grounding/ungrounded-seed]].
- Relationship to Evol-Instruct: same "rewrite-harder" spirit but with explicit crossover (recombining two parents) in addition to mutation.
