---
author: Cursor Team
created: '2026-05-18'
description: A substantial improvement in intelligence and behavior over Composer
  2, particularly on long-horizon agentic tasks.
kind: post
links:
  source: https://cursor.com/blog/composer-2-5
owner: blaz
read: false
slug: cursor-composer-2-5
source: Cursor
tags:
- type/post
- status/stub
title: Introducing Composer 2.5
type: note
updated: '2026-05-18'
year: 2026
---

# Introducing Composer 2.5

> *Cursor Team* — Cursor blog, 2026

## TL;DR

Cursor's Composer 2.5 coding model. Built on the same open-source checkpoint as
Composer 2 — Moonshot's Kimi K2.5. Gains attributed to three levers: scaled
training, more complex RL environments, and new learning methods.

Training-stack changes:

- **Targeted textual feedback** — a localized credit-assignment method. For a
  problematic model message, a short corrective hint is inserted into the local
  context; the resulting hinted distribution is used as a teacher and an
  on-policy distillation KL loss moves the (unhinted) student toward it. Gives a
  localized signal for behaviors (bad tool call, style violation) that a
  whole-rollout reward cannot localize.
- **Synthetic data** — trained with 25× more synthetic tasks than Composer 2,
  grounded in real codebases (e.g. *feature deletion*: delete testable features
  from a real repo, task is to reimplement, repo tests are the verifiable
  reward). Harder tasks are selected and created dynamically as the model
  saturates training problems. Large-scale synthetic task creation surfaced
  sophisticated reward hacking, caught with agentic monitoring.
- **Sharded Muon + dual-mesh HSDP** — continued pretraining with Muon and
  distributed Newton-Schulz orthogonalization at per-head / per-expert
  granularity; separate HSDP meshes for non-expert vs. expert weights.

## Notes

(stub) Synthetic-data / RL-environment material from this post is integrated
into the synthesis tier — see [[concepts/rl-environment-construction]] (task
generation, curriculum, reward-hacking pathologies).

## Source

- Original URL: <https://cursor.com/blog/composer-2-5>
