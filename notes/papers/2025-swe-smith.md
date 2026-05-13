---
arxiv: '2504.21798'
authors:
- '[needs verification]'
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.21798
  raw: null
  source: https://arxiv.org/abs/2504.21798
owner: blaz
read: false
slug: swe-smith
tags:
- type/paper
- source/primary
- status/stub
- domain/code
- domain/agents
- stage/sft
title: SWE-Smith
type: note
updated: '2026-05-10'
year: 2025
---

# SWE-Smith

- **arXiv**: [2504.21798](https://arxiv.org/abs/2504.21798)
- **Authors / affiliation**: John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, Diyi Yang (Stanford / Princeton / NYU)
- **Year / venue**: 2025
- **Grounding axis**: [[maps/grounding/repository-multifile]]
- **Output shape**: Synthesized task instances + agent trajectories
- **Filter / verification**: Execution-based — generated tasks must break existing tests; trajectories filtered by fail-to-pass
- **Training stage**: Trajectory-SFT

## Method

Automated pipeline for synthesizing SWE task instances from arbitrary Python codebases. Unlike SWE-Bench or SWE-Gym (which mine existing PR-based issues), SWE-Smith *generates* tasks by mutating a codebase until existing tests break, then wraps each broken state as a task specification.

Steps (per paper):
1. Take a Python codebase with an executable test suite.
2. Build a corresponding execution environment.
3. Auto-synthesize 100s–1,000s of task instances that break existing tests per repo.
4. Collect agent rollouts on the generated tasks; filter by fail-to-pass outcome.

Decoupling task supply from historical PRs is the key design move — it removes the "must have merged PRs with tests" constraint and enables much wider repo coverage.

## Dataset

- **50,000 task instances** sourced from **128 GitHub repositories** — ~10× larger than prior work (SWE-Gym: 2,438 from 11 repos).
- Open releases: pipeline code, task instances, trajectories, and trained models.

## Key result

SWE-agent-LM-32B trained on SWE-Smith data: **40.2%** resolve rate on SWE-Bench Verified — SOTA among open-weight models at time of release.

## Notes

- The scale gain (50K vs. 2.4K) comes from task synthesis, not repo breadth per se — ~400 tasks per repo on average.
- Peer of [[notes/papers/2025-swe-rl]] by benchmark number (40.2% vs. 41.0% on SWE-bench Verified), but methodologically orthogonal: SWE-Smith relies on execution-reward SFT over synthesized tasks; SWE-RL uses rule-based reward (diff similarity) over real historical PRs.
- Distinct from [[notes/papers/2025-r2e-gym]]: R2E-Gym synthesizes the *environments*; SWE-Smith synthesizes the *tasks* over real repos.
- Distinct from [[notes/papers/2024-swe-gym]]: SWE-Gym mines real PR-based issues; SWE-Smith generates tasks by breaking tests.
- Together with SWE-Gym, SWE-RL, and R2E-Gym, frames the trajectory-synthesis frontier: see [[concepts/trajectory-synthesis]].
- **Poolside internal eval**: [[evals/swe-smith-test-split]] (10% test split, pass_rate + pass_rate@k, Harbor harness)
