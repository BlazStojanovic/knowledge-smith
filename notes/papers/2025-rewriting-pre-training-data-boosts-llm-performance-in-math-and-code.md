---
arxiv: '2505.02881'
authors:
- Kazuki Fujii
- Yukito Tajima
- Sakae Mizuki
- Masaki Kawamura
- Hinari Shimada
- Taihei Shiotani
- Koshiro Saito
- Masanari Oi
- Taishi Nakamura
- Takumi Okamoto
- Shigeki Ishida
- Kakeru Hattori
- Youmi Ma
- Hiroya Takamura
- Rio Yokota
- Jun Sakuma
- Naoaki Okazaki
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.02881
  raw: '[[raw/papers/md/2025-rewriting-pre-training-data-boosts-llm-performance-in-math-and-code]]'
  source: null
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-rewriting-pre-training-data-boosts-llm-performance-in-math-and-code.md
raw_pdf: raw/papers/pdf/2025-rewriting-pre-training-data-boosts-llm-performance-in-math-and-code.pdf
read: false
slug: rewriting-pre-training-data-boosts-llm-performance-in-math-and-code
tags:
- type/paper
- status/draft
- source/primary
- confidential/public-source
- domain/synth-data
- domain/code
- domain/pretraining
- domain/data-mix
title: Rewriting Pre-Training Data Boosts LLM Performance in Math and Code
type: note
updated: '2026-05-15'
year: 2025
---

# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code

## Citation

- URL: https://arxiv.org/abs/2505.02881
- arXiv: 2505.02881 (v4, 2026-03-01; cs.LG / cs.CL)
- Authors: Kazuki Fujii, Yukito Tajima, Sakae Mizuki, et al. (Institute of Science Tokyo; AIST)
- Year: 2025
- PDF: [[raw/papers/pdf/2025-rewriting-pre-training-data-boosts-llm-performance-in-math-and-code.pdf]]

## TL;DR

Introduces **SwallowCode** (≈16.1B Python tokens) and **SwallowMath** (≈2.3B tokens), open datasets built by *rewriting* — not just filtering — public corpora. The "transform-and-retain" thesis: instead of discarding low-quality samples, rewrite them, maximising data utility. Under a fixed 50B-token continual-pretraining budget on Llama-3.1-8B, SwallowCode gives **+17.0 HumanEval / +16.1 HumanEval+** pass@1 over Stack-Edu; SwallowMath gives **+12.4 GSM8K / +7.6 MATH**. Datasets released under the Llama 3.3 Community License; prompts and checkpoints are public.

## Code pipeline (SwallowCode)

Four stages over `the-stack-v2-train-smol-ids` Python, each adopted only after an ablation showed it helped:

1. **Syntax-error filter** — `compile()` each sample, drop non-compiling. ≈41M → 37M samples (−9.7%).
2. **Linter filter** — pylint, keep score ≥ 7.0 (0–10), plus a heuristic comment-ratio penalty for over-commented / non-functional files. 36.7M → 24.1M (−34.3%).
3. **SGCR** (Style-Guided Code Rewriting) — LLM rewrite to the Google Python Style Guide across ten criteria (naming, docstrings, type hints, modularity, error handling, …). Meaning-preserving; condenses data (avg 836 → 548 tokens).
4. **SCOR** (Self-Contained Optimization Rewriting) — LLM rewrite for self-containment (inline/satisfy external deps), algorithmic efficiency, and upgrading trivial snippets into instructive examples. Meaning-extending; expands data (avg 548 → 835 tokens).

Rewriting LLM: **Llama-3.3-70B-Instruct**. SGCR and SCOR run as a *decoupled two-stage* pass — SCOR consumes SGCR's output. A merged single-pass prompt with all 19 directives caused **instruction drift** (the model satisfied a subset and dropped the rest), motivating the split. Contribution split: filtering ≈ +1–2 pts, SGCR +7–9, SCOR +5–6.

LLM-based quality *scoring* (a separate score-only filter) was ablated but **not adopted** — marginal gains (<1 pt) for ≈22% extra compute versus rewriting.

## Prompts

The verbatim SGCR / SCOR prompts (Appendix E.4 / E.5) and the math-rewriting prompt (Appendix F) are extracted and discussed in [[concepts/code-rewriting-prompts]]. NVIDIA's Nemotron-CC-Code reuses SGCR + SCOR (run with Qwen3-32B) as two of its three code-rewriting prompt families.

## Caveats from the paper

- Python-only; the pipeline is claimed language-agnostic but only validated on Python.
- SGCR's `snake_case` normalisation breaks MBPP's non-standard function names → MBPP excluded from the benchmark suite as a measurement artifact, not a true regression.
- Rewriting may propagate the rewriting LLM's stylistic biases; decontamination + self-contamination checks reported as clean (Appendix H).

## Relevance to Poolside

Canonical public recipe for [[projects/dat-404-raw-code-rephrasing]] — "rewrite raw code for pretraining". The transform-and-retain framing and the two-stage prompt design are direct reference points for the raw-code rephrasing work.

## Related notes

- [[concepts/code-rewriting-prompts]]
- [[concepts/rephrasal-operations]]
- [[projects/dat-404-raw-code-rephrasing]]

## Reading state

`read: false` — captured 2026-05-15; body summarised from the PDF, not yet marked read by Blaz.
