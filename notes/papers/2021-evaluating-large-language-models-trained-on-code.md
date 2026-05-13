---
arxiv: '2107.03374'
authors:
- Chen
- Tworek
- Jun
- Yuan
- Pinto de Oliveira
- Kaplan
- Edwards
- Burda
- Joseph
- Brockman
- Ray
- Puri
- Krueger
- Petrov
- Khlaaf
- Sastry
- Mishkin
- Chan
- Gray
- Ryder
- Pavlov
- Power
- Kaiser
- Bavarian
- Winter
- Tillet
- Such
- Cummings
- Plappert
- Chantzis
- Barnes
- Herbert-Voss
- Hebgen Guss
- Nichol
- Paino
- Tezak
- Tang
- Babuschkin
- Balaji
- Jain
- Saunders
- Hesse
- Carr
- Leike
- Achiam
- Misra
- Morikawa
- Radford
- Knight
- Brundage
- Murati
- Mayer
- Welinder
- McGrew
- Amodei
- McCandlish
- Sutskever
- Zaremba (OpenAI)
created: 2026-04-22
kind: paper
links:
  code: https://github.com/openai/human-eval
  paper: https://arxiv.org/abs/2107.03374
  raw: '[[raw/papers/md/2021-evaluating-large-language-models-trained-on-code]]'
  source: https://arxiv.org/abs/2107.03374
owner: blaz
raw_pdf: raw/papers/pdf/2021-evaluating-large-language-models-trained-on-code.pdf
read: false
slug: evaluating-large-language-models-trained-on-code
tags:
- type/paper
- status/draft
- domain/code
- domain/evals
- source/primary
title: Evaluating Large Language Models Trained on Code
type: note
updated: '2026-05-10'
year: 2021
---

# Evaluating Large Language Models Trained on Code

## Citation

- URL: https://arxiv.org/abs/2107.03374
- Authors: Chen, Tworek, Jun, Yuan, Pinto de Oliveira, Kaplan, Edwards, Burda, Joseph, Brockman, Ray, Puri, Krueger, Petrov, Khlaaf, Sastry, Mishkin, Chan, Gray, Ryder, Pavlov, Power, Kaiser, Bavarian, Winter, Tillet, Such, Cummings, Plappert, Chantzis, Barnes, Herbert-Voss, Hebgen Guss, Nichol, Paino, Tezak, Tang, Babuschkin, Balaji, Jain, Saunders, Hesse, Carr, Leike, Achiam, Misra, Morikawa, Radford, Knight, Brundage, Murati, Mayer, Welinder, McGrew, Amodei, McCandlish, Sutskever, Zaremba (OpenAI)
- Year / venue: 2021 / arXiv
- **Raw**: [[raw/papers/pdf/2021-evaluating-large-language-models-trained-on-code]]

## Core Claim

Introduces Codex (GPT fine-tuned on GitHub code) and the HumanEval benchmark — 164 hand-written Python problems for evaluating functional correctness via pass@k. Repeated sampling is surprisingly effective: Codex-12B solves 28.8% at pass@1 but 72.3% at pass@100.

## Key Paper Ideas

- **pass@k with unbiased estimator**: the standard metric for code generation evaluation. Generates n≥k samples, counts c correct, computes `pass@k = E[1 - C(n-c,k)/C(n,k)]`. The numerically stable implementation avoids overflow from large binomial coefficients. The biased alternative `1-(1-p_hat)^k` consistently underestimates even when n>5k.
- **Execution-based > match-based**: BLEU score distributions for correct and incorrect solutions overlap significantly, proving that BLEU optimization is not equivalent to functional correctness optimization. This is the definitive argument for execution-based code evaluation.
- **Temperature-k tradeoff**: optimal sampling temperature depends on k. Lower T for pass@1 (0.2 for Codex-679M), higher T for pass@100 (0.8). Higher temperatures produce more diverse samples, and pass@k rewards finding *any* correct solution.
- **HumanEval design**: 164 hand-written Python problems, not scraped from any public source (to avoid contamination). Docstring-to-code format. Average 7.7 unit tests per problem. Difficulty comparable to "simple software interview questions."
- **Mean log-probability ranking**: practical proxy for oracle selection when unit tests unavailable. Codex-12B: 44.5% with log-prob ranking vs 28.8% single sample vs 77.5% oracle.

## Methodology

**pass@k unbiased estimator** (numpy implementation from Figure 3):
```python
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```
Standard protocol: n=200 samples per problem, nucleus sampling top_p=0.95.

**HumanEval**: docstring → function body generation. Stop at `\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`. Execute in sandboxed environment (gVisor, eBPF firewall).

**BLEU comparison** (Figure 8): for each HumanEval problem, plot BLEU distributions of correct vs incorrect Codex-12B solutions against reference. Significant overlap demonstrates BLEU inadequacy.

## Key Results

| Model | pass@1 | pass@10 | pass@100 |
|---|---|---|---|
| GPT-Neo 2.7B | 6.41% | 11.27% | 21.37% |
| GPT-J 6B | 11.62% | 15.74% | 27.74% |
| Codex-300M | 13.17% | 20.37% | 36.27% |
| Codex-12B | 28.81% | 46.81% | 72.31% |
| Codex-S-12B | 37.7% | — | 77.5% (oracle) |

Test loss follows power law: `(N / 5.92e7)^(-0.13)`. Performance degrades exponentially with number of chained operations in docstring (~2-3x drop per additional component).

## Core Concepts

- [[concepts/pass-at-k-methodology]] — this paper defines the methodology
- [[maps/model-evaluation/code-eval-paradigms]] — establishes function-level execution-based eval as the paradigm
- [[concepts/benchmark-saturation]] — HumanEval has since saturated (frontier models >95%), motivating EvalPlus, BigCodeBench, etc.

## Relevance To Poolside

HumanEval is a foundational benchmark in Poolside's eval suite. The pass@k estimator and execution-based evaluation paradigm are used across all Poolside code evals (EvalPlus, BigCodeBench, LiveCodeBench, MultiPL-E). The temperature-k tradeoff informs sampling strategy for code evaluation. The BLEU inadequacy finding justifies Poolside's exclusive use of execution-based code evaluation.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- HumanEval is saturated — what is the next function-level benchmark? (EvalPlus augments tests, BigCodeBench increases difficulty)
- The exponential degradation with chained operations — does this scale with model size or is it intrinsic to autoregressive generation?
- pass@k requires running n samples — at what point does the cost of sampling dominate the eval budget?

## Related Notes

- Evals: [[evals/humaneval]], [[evals/evalplus]], [[evals/bigcodebench]], [[evals/livecodebench]]
- Papers: [[notes/papers/2023-evalplus]], [[notes/papers/2024-bigcodebench]], [[notes/papers/2023-swe-bench]]
- Concepts: [[concepts/pass-at-k-methodology]], [[concepts/benchmark-saturation]]

## Caveats

- HumanEval is Python-only, function-level only, 164 problems — limited scope
- Now saturated: frontier models achieve >95% pass@1
- Unit tests per problem (avg 7.7) are insufficient to verify full correctness — EvalPlus addresses this
- Docstring quality varies — some problems underspecified
- Sandbox requirements not standardized — different implementations may produce different results
