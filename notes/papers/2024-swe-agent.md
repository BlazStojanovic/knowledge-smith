---
arxiv: '2405.15793'
authors:
- John Yang
- Carlos E. Jimenez
- Alexander Wettig
- Kilian Lieret
- Shunyu Yao
- Karthik Narasimhan
- Ofir Press
created: 2026-04-22
kind: paper
links:
  code: https://github.com/princeton-nlp/SWE-agent
  paper: https://arxiv.org/abs/2405.15793
  raw: '[[raw/papers/md/2024-swe-agent]]'
  source: https://arxiv.org/abs/2405.15793
owner: blaz
raw_pdf: raw/papers/pdf/2024-swe-agent.pdf
read: false
slug: swe-agent
tags:
- type/paper
- source/primary
- status/draft
- domain/evals
- domain/agents
- domain/code
title: 'SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering'
type: note
updated: '2026-05-10'
year: 2024
---

# SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

Agent-computer interface (ACI) design — the commands, feedback formats, and guardrails exposed to an LM agent — matters as much as the underlying model for automated software engineering performance.

## Citation

- **Authors**: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press
- **Affiliation**: Princeton Language and Intelligence, Princeton University
- **Venue**: NeurIPS 2024
- **arXiv**: [2405.15793](https://arxiv.org/abs/2405.15793)
- **Raw PDF**: [[raw/papers/pdf/2024-swe-agent.pdf]]

## Core Contribution

Two claims:

1. **Concept of the Agent-Computer Interface (ACI).** LM agents are a new category of end user with distinct abilities and limitations. Interfaces designed for humans (shells, GUIs) are suboptimal for LMs; purpose-built ACIs improve performance without changing model weights.
2. **SWE-agent system.** An open-source agent (LM + ACI) for end-to-end software engineering. Achieves SOTA on SWE-bench and HumanEvalFix at time of publication.

## Key Method: Agent-Computer Interface Design

### Architecture

SWE-agent is a ReAct-style agent (thought + action per turn) built atop a Linux shell. The ACI is an abstraction layer between the LM and the computer, specifying:

- The **commands** available to the agent.
- The **format of environment feedback** returned to the agent.
- **History management** — how prior observations are compressed before being fed as context.

### ACI Components

| Component | Commands | Design rationale |
|---|---|---|
| **Search / navigation** | `find_file`, `search_file`, `search_dir` | Summarized results (max 50 hits); if exceeded, agent told to refine query. Avoids exhaustive iteration. |
| **File viewer** | `open`, `scroll_down`, `scroll_up`, `goto` | Shows 100-line window with line numbers, file path, total lines, omitted-line counts. |
| **File editor** | `edit <start> <end> <replacement>` | Single command replaces a range of lines. File viewer auto-updates after edit. Integrated **linter** blocks edits that introduce syntax errors and shows before/after diff, asking agent to retry. |
| **Context management** | (system-level) | Observations older than the last 5 turns collapsed to single-line summaries. Malformed generations trigger error + retry. "No output" placeholder when command produces no stdout. |

The agent also retains access to standard Linux commands when needed.

### ACI Design Principles

Four principles distilled from iterative development on SWE-bench dev set:

1. **Actions should be simple and easy to understand.** Few options, concise documentation — reduces need for demos or fine-tuning.
2. **Actions should be compact and efficient.** Important operations consolidated into few actions. Avoid multi-turn compositions for single logical operations.
3. **Environment feedback should be informative but concise.** Show effect of action + current state; suppress irrelevant detail. LMs have fixed context cost and are harmed by distraction.
4. **Guardrails mitigate error propagation and hasten recovery.** E.g., syntax checker on edits catches mistakes before they cascade.

### Key Claim: Interface Design >= Model Choice

The paper frames ACI design as analogous to HCI: just as good UIs make humans more productive, good ACIs make LMs more productive. The paper argues this is an underexplored axis — most prior work focused independently on tool use, prompting, or code execution, whereas ACI unifies these under a single interface-design framework.

## Key Results

### SWE-bench (full test set, 2,294 instances)

| System | Model | % Resolved | $ Avg. Cost |
|---|---|---|---|
| RAG | GPT-4 Turbo | 1.31 | 0.13 |
| RAG | Claude 3 Opus | 3.79 | 0.25 |
| Shell-only agent | GPT-4 Turbo | 11.00 (Lite only) | 1.46 |
| **SWE-agent** | **GPT-4 Turbo** | **12.47** | **1.59** |
| SWE-agent | Claude 3 Opus | 10.46 | 2.59 |

### SWE-bench Lite (300 instances)

| System | Model | % Resolved | $ Avg. Cost |
|---|---|---|---|
| RAG | GPT-4 Turbo | 2.67 | 0.13 |
| Shell-only agent | GPT-4 Turbo | 11.00 | 1.46 |
| **SWE-agent** | **GPT-4 Turbo** | **18.00** | **1.67** |
| SWE-agent | Claude 3 Opus | 13.00 | 2.18 |

SWE-agent with GPT-4 Turbo achieves a **64% relative increase** over Shell-only on SWE-bench Lite (18.00% vs 11.00%), demonstrating ACI value. Compared to RAG, SWE-agent is 8-13x more costly but yields 6.7x improved resolve rate.

### HumanEvalFix

| Model | Python | JS | Java |
|---|---|---|---|
| GPT-4 (non-agent) | 47.0 | 48.2 | 50.0 |
| WaveCoder-DS-6.7B | 57.9 | 52.4 | 57.3 |
| **SWE-agent w/ GPT-4 Turbo** | **87.7** | **89.7** | **87.9** |

### Ablation Results (SWE-bench Lite, % Resolved)

| Ablation | % Resolved | Delta |
|---|---|---|
| **SWE-agent (full)** | **18.0** | — |
| Edit w/o linting | 15.0 | -3.0 |
| No edit (bash only) | 10.3 | -7.7 |
| Summarized search (default) | 18.0 | — |
| Iterative search (UI-inspired) | 12.0 | -6.0 |
| No search tools | 15.7 | -2.3 |
| 30-line viewer window | 14.3 | -3.7 |
| 100-line viewer window (default) | 18.0 | — |
| Full file viewer | 12.7 | -5.3 |
| Last 5 observations (default) | 18.0 | — |
| Full history | 15.0 | -3.0 |
| Without demonstration | 16.3 | -1.7 |

Key ablation findings:
- **Edit interface is critical**: removing it drops 7.7 pp. Linting adds 3.0 pp.
- **Iterative search is worse than no search** (12.0% vs 15.7%): agents exhaustively iterate through results, burning budget/context.
- **File viewer window size matters**: 100 lines is best; too little (30) or too much (full file) both hurt.
- **Context compression helps**: full history costs 3.0 pp vs. collapsing older observations.

## Agent Behavior Analysis

- **Reproduce then localize**: agents typically start by writing reproduction code or searching for relevant files. Most common triple of actions: `create`, `edit`, `python`.
- **Edit-execute loops**: from turn 5 onward, `edit` and `python` dominate, with intermittent localization.
- **Editing remains hard**: 51.7% of trajectories (1,185/2,294) contain at least one failed edit. Recovery probability is 90.5% on first attempt but drops to 57.2% after one failed edit.
- **Succeed fast, fail slow**: successful runs finish with median cost $1.21 / 12 steps; unsuccessful runs average $2.52 / 21 steps. 93.0% of resolved instances submit before exhausting cost budget.
- **Failure modes** (n=248 unresolved on Lite): Incorrect Implementation 39.9%, Failed to Recover from Edit 23.4%, Overly Specific Implementation 12.1%, remaining split across localization failures, premature giving up, reproduction failures, timeout.

## Limitations

- Results are with GPT-4 Turbo and Claude 3 Opus only. Open-source models (Llama 3, DeepSeek Coder) performed poorly in the agent setting, partly due to context window limits.
- ACI was designed via manual inspection and grid search on SWE-bench dev set — design choices may be overfit to this benchmark and Python-only repositories.
- Per-instance cost budget capped at $4; runs exceeding this auto-submitted existing edits.
- Evaluation is on SWE-bench (Python-only, 12 repos). Generalization to other languages/repo structures not tested.
- The paper does not compare to multi-agent or pipeline-based approaches (these emerged later).
- Pass@1 only; no systematic analysis of pass@k scaling beyond a 6-run variance plot on Lite.

## Connections

- [[concepts/agentic-evaluation-methodology]] — SWE-agent is a foundational example of how the evaluation harness (interface/scaffolding) can dominate the measured capability of the underlying model.
- [[notes/papers/2023-swe-bench]] — SWE-bench is the primary evaluation benchmark; SWE-agent was co-developed by largely the same Princeton team.
- [[notes/papers/2024-ai-agents-that-matter]] — Raises concerns about confounding scaffolding with model capability in agent benchmarks; SWE-agent's results illustrate this directly (same model, different interface = large performance delta).
- [[maps/model-evaluation/landscape]] — SWE-agent and SWE-bench sit in the agentic / repository-level code evaluation quadrant.

## Related Papers

- Jimenez et al. (2024) — SWE-bench: the benchmark. arXiv 2310.06770.
- Yao et al. (2023) — ReAct: the reasoning + acting framework SWE-agent builds on.
- Yang et al. (2024) — InterCode: the interactive coding framework from which Shell-only baseline is adapted.
- Muennighoff et al. (2024) — OctoPack / HumanEvalFix: second evaluation benchmark used.
- Cooper et al. (2007) — *About Face 3*: HCI textbook that inspired the ACI design philosophy.

## Blaz Notes

-
