---
arxiv: '2605.14445'
authors:
- Runyuan He
- Qiuyang Mang
- Shang Zhou
- Kaiyuan Liu
- Hanchen Li
- Huanzhi Mao
- Qizheng Zhang
- Zerui Li
- Bo Peng
- Lufeng Cheng
- Tianfu Fu
- Yichuan Wang
- Wenhao Chai
- Jingbo Shang
- Alex Dimakis
- Joseph E. Gonzalez
- Alvin Cheung
created: '2026-05-19'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.14445
  raw: '[[raw/papers/md/2026-frontiersmith-synthesizing-open-ended-coding-problems]]'
  source: https://arxiv.org/abs/2605.14445
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-frontiersmith-synthesizing-open-ended-coding-problems.md
raw_pdf: raw/papers/pdf/2026-frontiersmith-synthesizing-open-ended-coding-problems.pdf
read: false
slug: frontiersmith-synthesizing-open-ended-coding-problems
tags:
- domain/synth-data
- domain/code
- synthetic-data
- open-ended
- competitive-programming
- data-synthesis
- type/paper
- status/draft
title: 'FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale'
type: note
updated: '2026-05-19'
year: 2026
---

# FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale

> *He, Mang, Zhou, Liu, Li, Mao, Zhang et al.* — arXiv preprint, 2026

## TL;DR

Closed-ended coding problems (competitive programming, where correctness is binary and verifiable) are abundant; **open-ended** ones (no certifiable optimum, continuous quality score) are scarce and expensive to hand-curate. FrontierSmith is an automated pipeline that **mutates closed-ended CP problems into open-ended variants**, then filters them by a novel **idea divergence** metric — the estimated probability that two independent solvers pick different core algorithms. Agents build test cases + a continuous-score verifier for survivors. Training Qwen3.5-9B/27B with GRPO on 200 synthesized problems gets within ~0.5 pt of human-curated open-ended data on FrontierCS and beats it on ALE-bench, while crushing the closed-ended seed corpus baseline. Key claims: (1) closed-ended problems work as seeds once mutations remove the known optimum; (2) idea divergence is a tractable, effective problem-quality filter; (3) synthetic problems elicit the same long-horizon agent behavior (100+ turns, millions of tokens) as human-curated ones.

## Abstract

Many real-world coding challenges are open-ended and admit no known optimal solution. Yet, recent progress in LLM coding has focused on well-defined tasks such as feature implementation, bug fixing, and competitive programming. Open-ended coding remains a weak spot for LLMs, largely because open-ended training problems are scarce and expensive to construct. Our goal is to synthesize open-ended coding problems at scale to train stronger LLM coders. We introduce FrontierSmith, an automated system for iteratively evolving open-ended problems from existing closed-ended coding tasks. Starting from competitive programming problems, FrontierSmith generates candidate open-ended variants by changing the problems' goals, restricting outputs, and generalizing inputs. It then uses a quantitative idea divergence metric to select problems that elicit genuinely diverse approaches from different solvers. Agents then generate test cases and verifiers for the surviving candidates. On two open-ended coding benchmarks, training on our synthesized data yields substantial gains over the base models: Qwen3.5-9B improves by +8.82 score on FrontierCS and +306.36 (Elo-rating-based performance) on ALE-bench; Qwen3.5-27B improves by +12.12 and +309.12, respectively. The synthesized problems also make agents take more turns and use more tokens, similar to human-curated ones, suggesting that closed-ended seeds can be a practical starting point for long-horizon coding data.

## Notes

### Problem framing

- **Closed vs. open-ended.** Closed-ended: discrete correctness, efficiently verifiable (CP, SWE-bench). Open-ended: no tractable certificate of optimality at target scale, submissions scored on continuous quality (e.g. cluster scheduling — feasibility easy, global optimality intractable).
- **Data asymmetry is the motivation.** Codeforces/LeetCode give hundreds of thousands of verified closed-ended problems; the two largest open-ended benchmarks, FrontierCS and ALE-bench, have only ~240 and ~40 human-curated problems. Manual construction needs a designed objective, a *continuous-score* verifier, reliable tests, and expert judgment that the problem isn't dominated by one strategy.
- Prior coding-data-synthesis work (AutoCode, SWE-smith, HardTests, rStar-Coder, CodeContests+, BugPilot, GASP, Absolute Zero) is all closed-ended / binary-correctness. None transfers to open-ended.

### Method — the FrontierSmith pipeline (Algorithm 1)

A problem formulation is a tuple **(O, C_I, C_O)**: computational goal, admissible inputs, output constraints. Five stages, run iteratively (validated problems re-enter the seed pool, so each round draws from an increasingly diverse set):

1. **Seed** — closed-ended CP problems (pool initialized with HardTests).
2. **Mutate** — LLM extracts (O, C_I, C_O), then applies one or more of three mutation types, each removing the known optimum while keeping a continuous quality measure:
   - **Change goals (O→O′):** swap a decision/exact-answer goal for an optimization one. E.g. 2-SAT (decide satisfiability) → Min-True 2-SAT (minimize true variables).
   - **Restrict outputs (C_O→C_O′):** tighten output constraints, goal fixed. E.g. minimum spanning tree (greedy) → degree-constrained spanning tree (NP-hard).
   - **Generalize inputs (C_I→C_I′):** relax structural assumptions. E.g. max independent set on bipartite graphs (poly) → on arbitrary graphs (NP-complete).
3. **Filter** — two stages:
   - *Coarse LLM-as-judge:* checks the candidate defines an optimization objective with no known optimum, admits multiple plausible strategies, and is meaningfully scorable. Validated separately: 9% false-positive rate on closed-ended inputs, 19% false-negative on FrontierCS (acceptable — tuned for precision over recall).
   - *Idea divergence filter:* see below.
4. **Build env** — a **test-case agent** writes generator programs (varying size/structure, adversarial inputs targeting specific strategies) and a **verifier agent** writes a scoring program V_c → [0,1] (Eq. 4: normalize objective against a baseline solution s*; crashes/timeouts → 0). The two agents **cross-validate** each other's output iteratively; candidates not converging in a fixed number of rounds are discarded. Only **~10%** of candidates entering this stage produce a validated (T_c, V_c) pair.
5. **Output** — re-rank survivors by execution-grounded divergence; keep top N_final.

### Idea divergence metric

Defined as **d(c) = P[strategy(s_i) ≠ strategy(s_j)]** for two independently sampled solutions — the probability two solvers use different *core algorithms*. Exact computation intractable; two complementary estimators:

- **LLM-based:** draw n solutions, an LLM-as-judge labels each pair same/different strategy; d̂ = fraction judged different. O(n²) calls batched into small groups. Captures *semantic* strategy differences (greedy vs. DP). Applied first — needs no test infra.
- **Execution-grounded:** after test infra exists, build score vector q_i over m test cases; d̂ = average pairwise √(1/m)·‖q_i − q_j‖₂. Captures *behavioral* differences (e.g. speed/accuracy trade-offs). Used to refine the ranking.

Rationale for the filter: if one strategy dominates, the problem degenerates to closed-ended; and under GRPO, varied strategies with meaningfully different rewards give a stronger gradient than samples following one heuristic.

### Experimental results

- **Setup:** GPT-5.4 Thinking for mutation/filter/LLM-divergence; Claude Sonnet 4.6 for solution sampling (n=10), test + verifier generation. 4 iterations, B=1000 seeds/iter, N_div=100, N_final=50 → 200 synthesized problems. Train Qwen3.5-9B/27B with veRL + GRPO, 100 steps, single-turn.
- **Benchmarks:** FrontierCS (172 algorithmic problems, 0–100 scale), ALE-bench-lite (10 AtCoder Heuristic Contest tasks, Elo-style rating).
- **Main result (Table 1, Avg@5):** FrontierSmith-9B = 10.62 FrontierCS / 633.58 ALE-bench, vs. base 1.80 / 327.22 — within 0.55 pt of human-curated FrontierCS data (11.17) and above its ALE-bench score (558.49). FrontierSmith-27B = 19.82 / 661.64, *beating* human-curated FrontierCS (13.98 / 543.80) on both.
- **Controls.** Training directly on the closed-ended HardTests seeds: only 5.38 / 397.18 (9B) — mutation is what makes the data effective. Random-reward control ≈ base, ruling out problem-format exposure as the source of gains (consistent with Shao et al. 2025 on spurious rewards).
- **Filter ablation.** Removing both filters ("No Filter") drops FrontierCS 10.62→8.57 (≈2 pt) and ALE-bench 633.6→564.4.
- **Divergence as classifier (Fig. 5).** LLM-based estimate cleanly separates open-ended sources (FrontierCS/FrontierSmith/ALE-bench ≈ 0.4) from closed-ended HardTests (0.14, ~3× lower). FrontierSmith (0.42) edges human-curated FrontierCS (0.40).
- **Long-horizon behavior.** Run through Claude SDK / Codex / Kimi Code agents, FrontierSmith problems push agents into the long-horizon regime (Claude SDK: 113 turns, 6.3×10⁶ tokens), matching ALE-bench; HardTests and FrontierCS stay short-horizon.

### Limitations (per the paper)

- Pipeline only covers **self-contained algorithmic environments** (text in → program out). Excludes repo-level open-ended tasks needing environment setup — cloud system optimization, GPU kernel tuning, multi-file SWE.
- RL limited to **100 steps of single-turn GRPO**; no agentic/multi-turn RL despite the long-horizon nature of the problems — flagged as the natural next step.

### Relevance to our work

- A concrete **rephrasing/mutation pattern** for synthetic data: structured formulation-level mutation (O, C_I, C_O) plus a quantitative diversity filter, rather than free-form prompt rephrasing — potentially relevant framing for `dat-404-raw-code-rephrasing`.
- **Idea divergence** extracted as a metric note: [[metrics/idea-divergence]] — placed in context of [[poolside-concepts/codeanvil]] / structured-web as a candidate RL-task routing signal.
- FrontierCS and ALE-bench are open-ended coding benchmarks not yet documented in `evals/`.
- Note the artifact provenance: this is a UC Berkeley / FrontierCS-team paper; mutation/judging done with GPT-5.4 and Claude Sonnet 4.6, so the synthetic data inherits those models' priors.

## Source

- Raw markdown: [[knowledge-smith/raw/papers/md/2026-frontiersmith-synthesizing-open-ended-coding-problems]]
- PDF: [[knowledge-smith/raw/papers/pdf/2026-frontiersmith-synthesizing-open-ended-coding-problems.pdf]]
- arXiv: <https://arxiv.org/abs/2605.14445>
