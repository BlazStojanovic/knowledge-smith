---
arxiv: '2210.03629'
authors:
- Shunyu Yao
- Jeffrey Zhao
- Dian Yu
- Nan Du
- Izhak Shafran
- Karthik Narasimhan
- Yuan Cao
created: 2026-04-22
kind: paper
links:
  code: https://react-lm.github.io/
  paper: https://arxiv.org/abs/2210.03629
  raw: '[[raw/papers/md/2022-react]]'
  source: https://arxiv.org/abs/2210.03629
owner: blaz
raw_pdf: raw/papers/pdf/2022-react.pdf
read: false
slug: react
tags:
- type/paper
- source/primary
- status/draft
- domain/agents
- domain/reasoning
title: 'ReAct: Synergizing Reasoning and Acting in Language Models'
type: note
updated: '2026-05-10'
year: 2022
---

# ReAct: Synergizing Reasoning and Acting in Language Models

LLMs interleave free-form reasoning traces with task-specific actions in a closed loop, grounding chain-of-thought in external observations and outperforming reasoning-only and action-only baselines across QA, fact verification, and interactive decision-making tasks.

## Citation

- **Authors**: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **Affiliations**: Princeton University, Google Research (Brain team)
- **Venue**: ICLR 2023
- **arXiv**: [2210.03629](https://arxiv.org/abs/2210.03629)
- **Raw PDF**: [[raw/papers/pdf/2022-react.pdf]]

## Core Contribution

ReAct (Reason + Act) augments an LLM agent's action space to include both task-specific actions **and** free-form language "thoughts" (reasoning traces). Thoughts do not affect the external environment; they compose useful information from the current context to guide future actions. Actions interact with external environments and produce observations that feed back into subsequent reasoning.

Key idea formalised: the agent's action space becomes A_hat = A ∪ L, where A is the original task action space and L is the space of natural-language reasoning traces.

## Method

### Interleaved reasoning and acting

- For knowledge-intensive tasks (HotpotQA, FEVER): dense thought-action-observation steps alternate strictly. Each thought decomposes questions, extracts information from observations, performs commonsense/arithmetic reasoning, guides search reformulation, or synthesises the final answer.
- For decision-making tasks (ALFWorld, WebShop): thoughts appear sparsely at key decision points. The model decides when to insert a thought vs. take an action.

### Prompting setup

- Base model: PaLM-540B (frozen, few-shot in-context learning). Also tested with GPT-3.
- Few-shot exemplars: 6 for HotpotQA, 3 for FEVER, 3 per task type for ALFWorld (6 types), 1 for WebShop.
- Each exemplar is a human-annotated trajectory of thoughts, actions, and observations.
- No special format, thought design, or example selection procedure.

### Comparison to CoT and Act-only

| Method | What it generates | Grounding |
|---|---|---|
| Standard | Answer only | None |
| CoT (Reason Only) | Reasoning trace then answer | Internal knowledge only |
| Act Only | Actions + observations | External environment only |
| **ReAct** | Interleaved thoughts + actions + observations | Both internal reasoning and external environment |

### Combining ReAct and CoT-SC

Two hybrid strategies to leverage both internal and external knowledge:

- **ReAct → CoT-SC**: when ReAct fails to return an answer within a step budget (7 for HotpotQA, 5 for FEVER), fall back to CoT with self-consistency (21 samples, temperature 0.7, majority vote).
- **CoT-SC → ReAct**: when the CoT-SC majority answer occurs fewer than n/2 times (low confidence), fall back to ReAct.

### Finetuning

Bootstrap approach: 3,000 correct ReAct trajectories used to finetune PaLM-8B and PaLM-62B. Finetuned ReAct outperforms all prompting methods at larger scale (PaLM-8B finetuned ReAct > all PaLM-62B prompting; PaLM-62B finetuned ReAct > all PaLM-540B prompting).

## Key Results

### Knowledge-intensive reasoning (PaLM-540B prompting)

| Method | HotpotQA (EM) | FEVER (Acc) |
|---|---|---|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC (21 samples) | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| **ReAct** | **27.4** | **60.9** |
| CoT-SC → ReAct | 34.2 | **64.6** |
| ReAct → CoT-SC | **35.1** | 62.0 |
| Supervised SoTA | 67.5 | 89.5 |

- ReAct outperforms Act on both tasks.
- ReAct outperforms CoT on FEVER (60.9 vs. 56.3); slightly lags on HotpotQA (27.4 vs. 29.4).
- Best prompting results come from combining ReAct with CoT-SC: 35.1 EM on HotpotQA (ReAct → CoT-SC), 64.6 Acc on FEVER (CoT-SC → ReAct).
- Combined methods reach CoT-SC (21 samples) performance with only 3-5 samples.

### Error analysis (HotpotQA, 200 manually labelled trajectories)

| Mode | ReAct | CoT |
|---|---|---|
| Success — true positive | 94% | 86% |
| Success — false positive (hallucinated) | 6% | 14% |
| Failure — reasoning error | 47% | 16% |
| Failure — search result error | 23% | — |
| Failure — hallucination | 0% | 56% |
| Failure — label ambiguity | 29% | 28% |

- CoT hallucination is the dominant failure mode (56% of errors). ReAct has 0% hallucination errors.
- ReAct's dominant failure modes: reasoning errors (47%, including repetitive-step loops) and uninformative search results (23%).

### Interactive decision making

**ALFWorld** (134 unseen evaluation games):

| Method | Overall success rate |
|---|---|
| BUTLER (best of 8, imitation learning, 10^5 expert trajectories) | 37% |
| Act (best of 6) | 45% |
| ReAct-IM (best of 6) | 53% |
| **ReAct (best of 6)** | **71%** |
| ReAct (average of 6) | 57% |

- ReAct outperforms BUTLER by 34 percentage points absolute.
- Even the worst ReAct trial (48%) beats the best Act (45%) and BUTLER (37%) trials.
- ReAct vs. Act relative gain across 6 controlled trials: 33%–90%, averaging 62%.

**WebShop** (500 test instructions):

| Method | Score | Success Rate |
|---|---|---|
| IL (1,012 trajectories) | 59.9 | 29.1 |
| IL+RL (10,587 instructions) | 62.4 | 28.7 |
| Act (1-shot) | 62.3 | 30.1 |
| **ReAct (1-shot)** | **66.6** | **40.0** |
| Human Expert | 82.1 | 59.6 |

- ReAct achieves 10 percentage point absolute improvement in success rate over prior best (IL+RL).
- 1-shot Act already matches IL/IL+RL trained on 10^3–10^4 examples.

## The Synergy Argument

Two directions of mutual benefit:

1. **Reason to act**: reasoning traces help the model decompose goals into subgoals, track progress, handle exceptions, adjust plans, and synthesise final answers. Without reasoning, Act agents fail to decompose goals or lose track of environment state.
2. **Act to reason**: actions retrieve external information that grounds subsequent reasoning in facts rather than hallucinated internal knowledge. CoT's static reasoning leads to hallucination (56% of CoT failures on HotpotQA); ReAct's access to external knowledge eliminates hallucination as a failure mode.

The interleaving also yields interpretability benefits: humans can inspect reasoning traces to understand the decision basis, distinguish model internal knowledge from externally retrieved facts, and even intervene by editing thoughts mid-trajectory.

## Limitations

- **Prompting budget**: complex tasks with large action spaces need more exemplars, which can exceed in-context length limits.
- **Reasoning errors and repetitive loops**: ReAct's structural constraint (interleaved thought-action-observation) reduces flexibility in formulating reasoning steps, leading to higher reasoning error rate than CoT (47% vs. 16% on HotpotQA). Repetitive action-thought loops are a specific failure mode.
- **Search quality dependency**: 23% of ReAct failures on HotpotQA stem from uninformative search results derailing reasoning.
- **Gap to supervised SoTA**: all prompting methods remain far from supervised state-of-the-art (e.g. 35.1 vs. 67.5 EM on HotpotQA).
- **Greedy decoding**: the authors suspect repetitive loops may partly result from greedy decoding; beam search might help.
- **Wikipedia API is weak**: the action space only retrieves small passage fragments by exact name, significantly weaker than standard retrievers.

## Connections

- [[concepts/agentic-evaluation-methodology]] — ReAct's thought-action-observation loop is a foundational pattern for agentic evaluation; the error taxonomy (hallucination, reasoning error, search error) is directly relevant to diagnosing agent failures.
- [[maps/model-evaluation/landscape]] — ReAct benchmarks (HotpotQA, FEVER, ALFWorld, WebShop) span knowledge-intensive QA, fact verification, and interactive decision-making, mapping across multiple evaluation categories.
- [[notes/papers/2024-ai-agents-that-matter]] — ReAct is one of the canonical agent paradigms that later work on agent evaluation methodology builds upon.

## Related Papers

- Wei et al., 2022 — Chain-of-thought prompting (CoT). ReAct extends CoT by adding grounded actions.
- Wang et al., 2022 — Self-consistency (CoT-SC). Combined with ReAct for best prompting results.
- Nakano et al., 2021 — WebGPT. Action-only web interaction without explicit reasoning traces.
- Huang et al., 2022b — Inner Monologue. Closest prior work for closed-loop reasoning + acting, but limited to environment-state feedback rather than flexible reasoning.
- Ahn et al., 2022 — SayCan. LLM-based robotic action planning without reasoning traces.
- Zelikman et al., 2022 — STaR. Bootstrapping rationales for finetuning; ReAct uses a similar bootstrap approach.
- Shridhar et al., 2020b — ALFWorld benchmark and BUTLER baseline.
- Yang et al., 2018 — HotpotQA benchmark.
- Thorne et al., 2018 — FEVER benchmark.
- Yao et al., 2022 — WebShop benchmark.

## Blaz Notes

-
