---
arxiv: '2404.06654'
authors:
- Cheng-Ping Hsieh
- Simeng Sun
- Samuel Kriman
- Shantanu Acharya
- Dima Rekesh
- Fei Jia
- Yang Zhang
- Boris Ginsburg
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.06654
  raw: '[[raw/papers/md/2024-ruler-what-s-the-real-context-size-of-your-long-context-language-models]]'
  source: https://arxiv.org/abs/2404.06654
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-ruler-what-s-the-real-context-size-of-your-long-context-language-models.md
raw_pdf: raw/papers/pdf/2024-ruler-what-s-the-real-context-size-of-your-long-context-language-models.pdf
read: false
slug: ruler-what-s-the-real-context-size-of-your-long-context-language-models
tags:
- type/paper
- status/stub
- long-context
- benchmark
- evaluation
- llm
title: 'RULER: What''s the Real Context Size of Your Long-Context Language Models?'
type: note
updated: '2026-05-18'
year: 2024
---

# RULER: What's the Real Context Size of Your Long-Context Language Models?

> *Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, et al.* — arXiv 2024

## TL;DR

RULER is a synthetic long-context benchmark built because vanilla needle-in-a-haystack (NIAH) only probes shallow retrieval. It extends NIAH with diverse needle types/quantities and adds two new task families — **multi-hop tracing** and **aggregation** — that test behaviours beyond search. 13 tasks, configurable sequence length and difficulty; 17 long-context LMs evaluated. Headline finding: near-perfect vanilla NIAH scores do not transfer — almost all models drop sharply as context grows, and only half of models claiming ≥32K context actually hold up at 32K. Even Yi-34B (200K claimed) leaves large headroom under longer inputs and harder tasks. Open-sourced; now a standard long-context eval. (Summary from abstract; note unread.)

## Abstract

The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the "needle") from long distractor texts (the "haystack"), has been widely adopted to evaluate long-context language models (LMs). However, this simple retrieval-based test is indicative of only a superficial form of long-context understanding. To provide a more comprehensive evaluation of long-context LMs, we create a new synthetic benchmark RULER with flexible configurations for customized sequence length and task complexity. RULER expands upon the vanilla NIAH test to encompass variations with diverse types and quantities of needles. Moreover, RULER introduces new task categories multi-hop tracing and aggregation to test behaviors beyond searching from context. We evaluate 17 long-context LMs with 13 representative tasks in RULER. Despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases. While these models all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K. Our analysis of Yi-34B, which supports context length of 200K, reveals large room for improvement as we increase input length and task complexity. We open source RULER to spur comprehensive evaluation of long-context LMs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2404.06654>
- PDF: [[raw/papers/pdf/2024-ruler-what-s-the-real-context-size-of-your-long-context-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2024-ruler-what-s-the-real-context-size-of-your-long-context-language-models]]
