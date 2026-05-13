---
arxiv: '2602.04942'
authors:
- Emiliano Penaloza
- Dheeraj Vattikonda
- Nicolas Gontier
- Alexandre Lacoste
- Laurent Charlin
- Massimo Caccia
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.04942
  raw: '[[raw/papers/md/2026-privileged-information-distillation-for-language-models]]'
  source: https://arxiv.org/abs/2602.04942
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-privileged-information-distillation-for-language-models.md
raw_pdf: raw/papers/pdf/2026-privileged-information-distillation-for-language-models.pdf
read: false
slug: privileged-information-distillation-for-language-models
tags:
- type/paper
- status/stub
title: Privileged Information Distillation for Language Models
type: note
updated: '2026-05-11'
year: 2026
---

# Privileged Information Distillation for Language Models

> *Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Training-time privileged information (PI) can enable language models to succeed on tasks they would otherwise fail, making it a powerful tool for reinforcement learning in hard, long-horizon settings. However, transferring capabilities learned with PI to policies that must act without it at inference time remains a fundamental challenge. We study this problem in the context of distilling frontier models for multi-turn agentic environments, which typically hide their internal reasoning and expose only action trajectories. This breaks standard distillation pipelines, since successful behavior is observable, but the reasoning process is not. For this, we introduce π-Distill, a joint teacher-student objective that trains a PI-conditioned teacher and an unconditioned student simultaneously using the same model. Additionally, we also introduce On-Policy Self-Distillation (OPSD), an alternative approach that trains using Reinforcement Learning (RL) with a reverse KL-penalty between the student and the PI-conditioned teacher. We show that both of these algorithms effectively distill frontier agents using action-only PI. Specifically, we find that π-Distill and, in some cases, OPSD, outperform industry standard practices (Supervised finetuning followed by RL) that assume access to full Chain-of-Thought supervision across multiple agentic benchmarks, models, and forms of PI. We complement our results with extensive analysis that characterizes the factors enabling effective learning with PI, focusing primarily on π-Distill and characterizing when OPSD is competitive.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.04942>
- PDF: [[raw/papers/pdf/2026-privileged-information-distillation-for-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2026-privileged-information-distillation-for-language-models]]
