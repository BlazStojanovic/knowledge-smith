---
aliases:
- Accumulate vs Replace
- Gerstgrasser 2024
- Is Model Collapse Inevitable
arxiv: '2404.01413'
authors:
- Matthias Gerstgrasser
- Rylan Schaeffer
- Apratim Dey
- Rafael Rafailov
- Henry Sleight
- John Hughes
- Tomasz Korbak
- Rajashree Agrawal
- Dhruv Pai
- Andrey Gromov
- Daniel A. Roberts
- Diyi Yang
- David L. Donoho
- Sanmi Koyejo
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.01413
  raw: https://arxiv.org/pdf/2404.01413
  source: https://arxiv.org/abs/2404.01413
owner: blaz
read: false
slug: is-model-collapse-inevitable-accumulate-vs-replace
tags:
- type/paper
- source/primary
- domain/synth-data
- status/stub
title: Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating
  Real and Synthetic Data (Gerstgrasser et al. 2024)
type: note
updated: '2026-05-10'
year: 2024
---

# Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data (Gerstgrasser et al. 2024)

## Citation

- arXiv: [2404.01413](https://arxiv.org/abs/2404.01413)
- Authors: Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, Sanmi Koyejo.
- Year / venue: arXiv 2024.

> [!warning] unverified
> Author list and exact title transcribed from arXiv listing; verify before quoting in external work. Earlier vault references called this "Tirumala et al." — that may have been an error / different first-author convention; treat author attribution as `?` until verified.

## Core claim (stub)

The **positive result** in the iteration-dynamics literature: model collapse from training on synthetic data is *not* inevitable. If real and synthetic data are *accumulated* (each round adds synthetic to a growing pool that retains the original real data) rather than *replaced* (each round trains only on the latest synthetic batch), tail-mass loss is bounded and downstream performance does not degrade arbitrarily. Reframes Curse-of-Recursion as a **regime** (replacement) rather than an inevitability.

## Why it's load-bearing

The third pillar of the iteration-dynamics taxonomy in [[concepts/iteration-dynamics]] (the "accumulation" failure-mode-that-isn't-a-failure). Without this result the literature is just "model collapse is bad"; with it, the practical question becomes "what mixing ratio of real to synthetic is safe."

## Status

Stub. Full read deferred. Filed because the vault references this result in multiple places without a paper note. Particularly worth a deep read when designing a self-improvement loop for any Poolside pipeline.
