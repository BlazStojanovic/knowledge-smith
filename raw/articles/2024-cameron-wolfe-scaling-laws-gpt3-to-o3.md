---
source: article
url: https://cameronrwolfe.substack.com/p/llm-scaling-laws
retrieved: 2026-04-23
title: "Scaling Laws for LLMs: From GPT-3 to o3"
author: Cameron R. Wolfe
publication: Substack (Deep (Learning) Focus)
date: 2024
license: excerpt-only
note: "Restrictive license — excerpted key points only, not full text."
---

# Scaling Laws for LLMs: From GPT-3 to o3

Author: Cameron R. Wolfe. Substack.

> [!warning] Excerpt only
> Substack has restrictive terms. Key arguments summarized below; read the original for full treatment.

## Key topics covered

1. **Three eras of scaling**: pre-training scaling (Kaplan → Chinchilla), post-training scaling (SFT, RLHF, DPO), test-time compute scaling (o1, o3, search + verification)
2. **Kaplan → Chinchilla transition**: from "bigger models" to "matched scaling" of N and D
3. **Post-training as a new scaling axis**: RLHF and SFT provide diminishing but real improvements on top of pretrain scaling
4. **Test-time compute**: repeated sampling, chain-of-thought, verification as a way to trade inference compute for model size
5. **Practical implications**: each era shifts where the bottleneck lies — from compute to data to inference cost

## Relevance

Good accessible overview tracing the full evolution of scaling law thinking. Useful for framing the [[maps/scaling-laws/landscape]] across all three eras.
