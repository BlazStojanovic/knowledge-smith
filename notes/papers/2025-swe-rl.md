---
arxiv: '2502.18449'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.18449
  raw: null
  source: https://arxiv.org/abs/2502.18449
owner: blaz
raw_pdf: raw/papers/pdf/2025-swe-rl.pdf
read: false
slug: swe-rl
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- domain/agents
- stage/rl-rule
title: SWE-RL
type: note
updated: '2026-05-10'
year: 2025
---

# SWE-RL

- **arXiv**: [2502.18449](https://arxiv.org/abs/2502.18449)
- **Authors / affiliation**: Yuxiang Wei et al. (Meta FAIR / facebookresearch)
- **Year / venue**: 2025, NeurIPS 2025
- **Raw**: [[raw/papers/pdf/2025-swe-rl]]
- **Grounding axis**: [[maps/grounding/repository-multifile]]
- **Output shape**: RL rollouts over PR-resolution tasks (trajectory + proposed diff)
- **Filter / verification**: **Rule-based reward** — similarity between agent-proposed diff and oracle merged-PR diff. No test harness required.
- **Training stage**: RL on top of Llama-3-70B

## Method

Trains on open-source **software evolution** data: code snapshots plus code changes plus events (issues, PRs, reviews), mined at scale from GitHub. The agent proposes a diff to resolve an issue; the reward is the similarity between the proposed diff and the oracle merged-PR diff.

Key properties:

- No proprietary models in the pipeline. No distilled teacher.
- No per-repo test harness required — reward is computed over the diff directly.
- RL is the training method; no SFT warm-start from distilled data is required.

## Key result

Llama3-SWE-RL-70B achieves **41.0%** on SWE-bench Verified — best reported for <100B models at release time. Reported generalisation to out-of-domain tasks: function-level coding, library use, code reasoning, mathematics, general language understanding. I.e. the RL signal induces reasoning transfer despite training only on software evolution data.

## Notes

- This is the canonical example in the "rule-based-reward" branch of [[concepts/trajectory-synthesis]]. Contrast with [[notes/papers/2024-swe-gym]] (execution reward) — they reach broadly comparable SWE-bench numbers but differ on transfer properties.
- Open question: whether the reported reasoning transfer is specific to the diff-similarity reward, or a general property of RL over software-evolution data.
- Code: [github.com/facebookresearch/swe-rl](https://github.com/facebookresearch/swe-rl).
