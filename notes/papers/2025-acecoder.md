---
arxiv: '2502.01718'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.01718
  raw: null
  source: https://arxiv.org/abs/2502.01718
owner: blaz
raw_pdf: raw/papers/pdf/2025-acecoder.pdf
read: false
slug: acecoder
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- stage/rl-exec
title: AceCoder
type: note
updated: '2026-05-10'
year: 2025
---

# AceCoder

- **arXiv**: [2502.01718](https://arxiv.org/abs/2502.01718)
- **Authors / affiliation**: Huaye Zeng et al. (TIGER-AI-Lab)
- **Year / venue**: 2025, ACL 2025
- **Raw**: [[raw/papers/pdf/2025-acecoder]]
- **Grounding axis**: [[maps/grounding/execution-traces]] (with test-first synthesis — links [[maps/grounding/test-first-synthesis]])
- **Output shape**: (prompt, code, test-pass-rate) tuples → preference pairs for reward modelling
- **Filter / verification**: Execution against synthesised tests; tests themselves filtered by agreement on a reference solution.
- **Training stage**: Reward-model training + RL (R1-style)

## Method

Pipeline:

1. **Test-case synthesis at scale** — LLM generates extensive (question, test-cases) pairs from existing code data.
2. **Preference pair construction** — for each question, sample multiple code candidates; rank them by fraction of tests passed; emit preference pairs $(y_a \succ y_b)$ when $R(x, y_a) > R(x, y_b)$ with $R$ the test-pass fraction.
3. **Reward model** — trained with Bradley-Terry loss on the preference pairs.
4. **RL** — R1-style RL starting from Qwen2.5-Coder-base using the reward model.

## Key result

- Best-of-32 sampling: Llama-3.1-8B-Ins +10 points average; Qwen2.5-Coder-7B-Ins +5 points; 7B model reported on par with 236B DeepSeek-V2.5.
- R1-style RL starting from Qwen2.5-Coder-base: +25% on HumanEval+ and +6% on MBPP+ after ~80 optimisation steps.

## Notes

- Exemplifies the **synthesised-test-first then train** recipe. Test quality is the critical variable; paper discusses test filtering via agreement.
- Connects [[maps/grounding/test-first-synthesis]] (tests synthesised first) and [[maps/grounding/execution-traces]] (execution is the reward signal).
- Code: [github.com/TIGER-AI-Lab/AceCoder](https://github.com/TIGER-AI-Lab/AceCoder).
