---
arxiv: '2601.16206'
authors:
- Daixuan Cheng
- Shaohan Huang
- Yuxian Gu
- et al. (Furu Wei senior)
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.16206
  raw: null
  source: https://arxiv.org/abs/2601.16206
owner: blaz
read: false
slug: llm-in-sandbox
tags:
- type/paper
- source/primary
- status/stub
- domain/agents
- domain/reasoning
- domain/training
title: Computer Environments Elicit General Agentic Intelligence in LLMs
type: note
updated: '2026-05-10'
year: 2026
---

# Computer Environments Elicit General Agentic Intelligence in LLMs

## Citation

- URL: https://arxiv.org/abs/2601.16206
- Authors: Daixuan Cheng, Shaohan Huang, Yuxian Gu, et al. (Furu Wei senior)
- Year / venue: 2026 / arXiv
- arXiv: [2601.16206](https://arxiv.org/abs/2601.16206)

## Core Claim

A minimal code-sandbox environment (LLM-in-Sandbox) elicits general "meta-capabilities" (resource access, file management, code execution) without additional training, yielding up to 15.5% gains across math, physics, chemistry, biomedicine, long-context, and instruction following, while reducing token consumption up to 8x.

## Key Paper Ideas

- **LLM-in-Sandbox**: virtualises computer as minimal sandbox with basic functionalities. At inference time, the model can write and execute code, access files, and manage resources — no fine-tuning needed.
- **LLM-in-Sandbox-RL**: training exclusively on non-agentic data within the sandbox lets weaker models internalise sandbox interactions. Generalises to unseen domains.
- Key insight: the sandbox environment itself provides a general-purpose reward signal — code either runs or it doesn't — making it a natural fit for [[concepts/rlvr|RLVR]].

## Key Results

- 15.5% average improvement across math, physics, chemistry, biomedicine, long-context, and instruction following.
- Up to 8× reduction in token consumption.
- RL variant (LLM-in-Sandbox-RL) improves weaker models without domain-specific data.

## Core Concepts

- Existing concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rlvr]], [[concepts/rl-environment-construction]]
- Concepts to extract: sandbox as general-purpose RL environment, meta-capabilities from environment interaction

## Relevance To Poolside

*Our interpretation*: demonstrates that a simple, general-purpose sandbox environment can serve as a universal RL training ground — the model doesn't need domain-specific environments for each capability. This is cheaper and more scalable than building per-domain sandboxes. Potentially relevant for extending Poolside's RL beyond code/math.

## Related Notes

- Concepts: [[concepts/rl-environment-construction]], [[concepts/rl-training-frameworks]]
- Maps: [[maps/rl-environments/landscape]]
