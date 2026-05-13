---
arxiv: '2505.20411'
authors:
- Ibragim Badertdinov
- Alexander Golubev
- Maksim Nekrashevich
- Anton Shevtsov
- Simon Karasik
- Andrei Andriushchenko
- Maria Trofimova
- Daria Litvintseva
- Boris Yangel
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.20411
  raw: '[[raw/papers/md/2025-swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated]]'
  source: https://arxiv.org/abs/2505.20411
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated.md
raw_pdf: raw/papers/pdf/2025-swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated.pdf
read: false
slug: swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated
tags:
- type/paper
- status/stub
title: 'SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated
  Evaluation of Software Engineering Agents'
type: note
updated: '2026-05-11'
year: 2025
---

# SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents

> *Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

LLM-based agents have shown promising capabilities in a growing range of software engineering (SWE) tasks. However, advancing this field faces two critical challenges. First, high-quality training data is scarce, especially data that reflects real-world SWE scenarios, where agents must interact with development environments, execute code and adapt behavior based on the outcomes of their actions. Existing datasets are either limited to one-shot code generation or comprise small, manually curated collections of interactive tasks, lacking both scale and diversity. Second, the lack of fresh interactive SWE tasks affects evaluation of rapidly improving models, as static benchmarks quickly become outdated due to contamination issues. To address these limitations, we introduce a novel, automated, and scalable pipeline to continuously extract real-world interactive SWE tasks from diverse GitHub repositories. Using this pipeline, we construct SWE-rebench, a public dataset comprising over 21,000 interactive Python-based SWE tasks, suitable for reinforcement learning of SWE agents at scale. Additionally, we use continuous supply of fresh tasks collected using SWE-rebench methodology to build a contamination-free benchmark for agentic software engineering. We compare results of various LLMs on this benchmark to results on SWE-bench Verified and show that performance of some language models might be inflated due to contamination issues.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2505.20411>
- PDF: [[raw/papers/pdf/2025-swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated.pdf]]
- Raw markdown: [[raw/papers/md/2025-swe-rebench-an-automated-pipeline-for-task-collection-and-decontaminated]]
