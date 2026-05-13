---
arxiv: '2605.03546'
authors:
- John Yang
- Kilian Lieret
- Jeffrey Ma
- Parth Thakkar
- Dmitrii Pedchenko
- Sten Sootla
- Emily McMilin
- Pengcheng Yin
- Rui Hou
- Gabriel Synnaeve
- Diyi Yang
- Ofir Press
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.03546
  raw: '[[raw/papers/md/2026-programbench-can-language-models-rebuild-programs-from-scratch]]'
  source: https://arxiv.org/abs/2605.03546
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-programbench-can-language-models-rebuild-programs-from-scratch.md
raw_pdf: raw/papers/pdf/2026-programbench-can-language-models-rebuild-programs-from-scratch.pdf
read: false
slug: programbench-can-language-models-rebuild-programs-from-scratch
tags:
- type/paper
- status/stub
title: 'ProgramBench: Can Language Models Rebuild Programs From Scratch?'
type: note
updated: '2026-05-11'
year: 2026
---

# ProgramBench: Can Language Models Rebuild Programs From Scratch?

> *John Yang, Kilian Lieret, Jeffrey Ma, Parth Thakkar, Dmitrii Pedchenko, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Turning ideas into full software projects from scratch has become a popular use case for language models. Agents are being deployed to seed, maintain, and grow codebases over extended periods with minimal human oversight. Such settings require models to make high-level software architecture decisions. However, existing benchmarks measure focused, limited tasks such as fixing a single bug or developing a single, specified feature. We therefore introduce ProgramBench to measure the ability of software engineering agents to develop software holisitically. In ProgramBench, given only a program and its documentation, agents must architect and implement a codebase that matches the reference executable's behavior. End-to-end behavioral tests are generated via agent-driven fuzzing, enabling evaluation without prescribing implementation structure. Our 200 tasks range from compact CLI tools to widely used software such as FFmpeg, SQLite, and the PHP interpreter. We evaluate 9 LMs and find that none fully resolve any task, with the best model passing 95\% of tests on only 3\% of tasks. Models favor monolithic, single-file implementations that diverge sharply from human-written code.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.03546>
- PDF: [[raw/papers/pdf/2026-programbench-can-language-models-rebuild-programs-from-scratch.pdf]]
- Raw markdown: [[raw/papers/md/2026-programbench-can-language-models-rebuild-programs-from-scratch]]
