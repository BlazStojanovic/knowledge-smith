---
arxiv: '2310.06770'
authors:
- Jimenez*
- Yang*
- Wettig
- Yao
- Pei
- Press
- Narasimhan (Princeton University
- University of Chicago)
created: 2026-04-22
kind: paper
links:
  code: https://github.com/princeton-nlp/SWE-bench
  paper: https://arxiv.org/abs/2310.06770
  raw: '[[raw/papers/md/2023-swe-bench]]'
  source: https://arxiv.org/abs/2310.06770
owner: blaz
raw_pdf: raw/papers/pdf/2023-swe-bench.pdf
read: false
slug: swe-bench
tags:
- type/paper
- status/draft
- domain/evals
- domain/code
- domain/agents
- source/primary
title: 'SWE-bench: Can Language Models Resolve Real-World GitHub Issues?'
type: note
updated: '2026-05-10'
year: 2023
---

# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

## Citation

- URL: https://arxiv.org/abs/2310.06770
- Authors: Jimenez*, Yang*, Wettig, Yao, Pei, Press, Narasimhan (Princeton University, University of Chicago)
- Year / venue: 2023 / ICLR 2024
- **Raw**: [[raw/papers/pdf/2023-swe-bench]]

## Core Claim

Introduces SWE-bench: 2,294 software engineering tasks from real GitHub issues and PRs across 12 Python repositories. The best model (Claude 2) solves only 1.96% of issues, establishing a challenging benchmark for repository-scale code editing that moves evaluation beyond function-level synthesis.

## Key Paper Ideas

- **Repository-scale evaluation paradigm**: moves from function-level synthesis (HumanEval, 164 isolated problems) to real-world issue resolution requiring multi-file, multi-function edits across codebases of 100K-900K lines
- **Fail-to-pass test methodology**: uses the transition of existing repo tests from FAIL→PASS as correctness signal, plus PASS→PASS regression checks. Grounds evaluation in real developer workflows without requiring hand-crafted tests.
- **Automated, refreshable pipeline**: 3-stage scrape→filter→validate pipeline applicable to any Python repo, enabling temporal freshness (built-in contamination resistance)
- **SWE-bench Lite as a design pattern**: 300-instance tractable subset for cost-effective evaluation — pattern adopted by Verified (500 instances)
- **Retrieval as evaluation variable**: BM25 vs oracle retrieval comparison separates localization ability from generation ability

## Methodology

**Construction pipeline**:
1. **Scrape**: top 100 most-downloaded PyPI packages → 12 popular repos → ~90,000 PRs
2. **Attribute filter** (→11,407): PR merged, resolves GitHub issue (regex: "fixes/closes/resolves #N"), changes test files
3. **Execution filter** (→2,294): apply test patch + solution, require ≥1 test transitions FAIL→PASS, filter install/runtime errors

**Task instance**: model receives issue text + codebase at base_commit, generates a .patch file. Evaluation applies test_patch + prediction_patch, checks FAIL_TO_PASS all pass AND PASS_TO_PASS all pass.

**Environment**: per-version conda environments with repo-specific installation, repository mirrors under SWE-bench GitHub org.

**Scale**: mean issue length 195 words, mean codebase 3,010 files / 438K lines, mean gold patch 32.8 lines across 1.7 files editing 3.0 functions. Mean 9.1 fail-to-pass tests, 120.8 total tests.

## Key Results

| Model | % Resolved (full) | % Resolved (Lite) |
|---|---|---|
| Claude 3 Opus | 3.79 | 4.33 |
| Claude 2 | 1.97 | 3.00 |
| GPT-4 Turbo | 1.31 | 2.67 |
| ChatGPT-3.5 | 0.17 | 0.33 |
| SWE-Llama 13B | 0.70 | 1.00 |

Oracle retrieval boosts Claude 2 from 1.97% to 4.8% (collapsed oracle: 5.93%). BM25 retrieves a superset of oracle files in only ~40% of instances. Models generate simpler patches (19.6 lines) than gold (44.1 lines), rarely editing >1 file.

## Core Concepts

- [[concepts/agentic-evaluation-methodology]] — SWE-bench defines the canonical agentic code eval pattern
- [[concepts/pass-at-k-methodology]] — SWE-bench uses binary resolution rather than pass@k, contrasting with HumanEval
- [[concepts/temporal-benchmark-robustness]] — refreshable pipeline provides temporal contamination resistance

## Relevance To Poolside

SWE-bench (and its Verified/Pro derivatives) is the primary agentic code evaluation benchmark in Poolside's Harbor harness. The fail-to-pass methodology, sandbox design, and Lite/Verified subset patterns directly shape how Poolside evaluates its coding agent. The retrieval-vs-generation decomposition is relevant to agent architecture choices.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- Does the pipeline generalize to non-Python repos? (SWE-bench Pro adds Go, TypeScript, JavaScript)
- The binary resolution metric (pass/fail) loses information — is there a useful partial-credit metric?
- At what point does model performance require harder instances (SWE-bench → Pro → harder)?

## Related Notes

- Evals: [[evals/swe-bench-verified-harbor]], [[evals/swe-bench-pro]], [[evals/swe-bench-agentless]], [[evals/swe-bench-agentless-multilingual]], [[evals/swe-polybench]], [[evals/multi-swe-bench]]
- Papers: [[notes/papers/2021-evaluating-large-language-models-trained-on-code]], [[notes/papers/2024-ai-agents-that-matter]]
- Concepts: [[concepts/agentic-evaluation-methodology]]

## Caveats

- Python only (12 repos) — does not cover multi-language evaluation
- Metric is binary per instance — no partial credit
- Generated code can pass tests while being suboptimal (less efficient, less readable)
- Environment setup is laborious (per-version conda environments, manual effort per repo)
- 32% of matplotlib issues contain images — requires multimodal capabilities
- Baselines only — paper establishes the benchmark, not the agent approach
