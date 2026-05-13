---
arxiv: '2409.16341'
authors:
- Shadi Iskander
- Nachshon Cohen
- Zohar Karnin
- Ori Shapira
- Sofia Tolmach
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2409.16341
  raw: '[[raw/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms.pdf]]'
  source: https://arxiv.org/abs/2409.16341
owner: blaz
raw_pdf: raw/papers/pdf/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms.pdf
read: true
slug: quality-matters-evaluating-synthetic-data-for-tool-using-llms
tags:
- type/paper
- status/draft
- source/paper
- confidential/public-source
- domain/llm
- domain/synth-data
- domain/evals
- domain/agents
title: 'Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs'
type: note
updated: '2026-05-10'
year: 2024
---

# Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs

## Citation

- URL: https://arxiv.org/abs/2409.16341
- PDF: https://arxiv.org/pdf/2409.16341
- Authors: Shadi Iskander, Nachshon Cohen, Zohar Karnin, Ori Shapira, Sofia Tolmach
- Year / venue: 2024-09-24 arXiv preprint
- arXiv: 2409.16341v2
- Categories: cs.LG, cs.CL, cs.SE
- Raw PDF: [[raw/papers/pdf/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms.pdf]]

## Core Claim

Existing synthetic datasets for tool-using LLMs ([[notes/papers/2024-toolbench|ToolBench]], [[notes/papers/2023-toolalpaca|ToolAlpaca]]) contain massive quality problems:
* 84% of [[notes/papers/2024-toolbench|ToolBench]] instances have at least one error. Filtering to a high-quality subset (~14% of the data) matches or beats training on the full dataset. 

The paper proposes two complementary evaluation methods: 
* a six-criterion intrinsic quality framework (automated via ChatGPT)
* In-Context Evaluation (ICE), a cheaper proxy that measures an instance's educational value as a one-shot example.

## Key Paper Ideas

1. **Six quality dimensions** split across instruction properties (**Specificity**, **Coherence**, **Solvability**) and API-call sequence properties (**Parameter Alignment**, **Sufficiency**, **Minimality**). These are instance-level, not dataset-level.

2. **Automated metrics via task reformulation.** Direct "is this specific?" prompting fails. Instead: model Specificity as parameter extraction (mark /#missing for gaps), Coherence as next-sentence prediction between instruction sentences, Parameter Alignment as extract-then-compare. These task-based prompts beat direct assessment by 10-20% accuracy.

3. **In-Context Evaluation (ICE).** Evaluate a training instance by how well it works as a one-shot example: plug it into a fixed prompt (10 API docs + 7 test queries), run a weak model, measure how much the instance helps via Levenshtein similarity to gold sequences. If showing instance x to LLaMA-7B helps it answer tool-use queries, x is probably good training data. Motivated by Dai et al. 2023 (ICL implicitly performs gradient descent → good in-context example ≈ good fine-tuning example). Cheap to compute, correlates with human criteria but captures something different — teaching effectiveness rather than correctness.

4. **Less-is-more for tool data.** 10K high-quality [[notes/papers/2024-toolbench|ToolBench]] instances (14% of data) achieve 0.54 pass rate vs. 0.45 for the full 73K. The effect is weaker on [[notes/papers/2023-toolalpaca|ToolAlpaca]] (already ~55% clean), confirming that filtering helps most when baseline quality is low.

## Methodology

### Intrinsic Quality Framework

Six binary criteria per instance:

| Dimension           | Type        | What it checks                                                                         |
| ------------------- | ----------- | -------------------------------------------------------------------------------------- |
| Specificity         | Instruction | All required details present for the LLM to fulfill the request                        |
| Coherence           | Instruction | Requests logically related, order makes real-world sense                               |
| Solvability         | Instruction | Requests addressable by the given API tools                                            |
| Parameter Alignment | Sequence    | Parameter values correctly extracted/inferred from instruction, no hallucinated values |
| Sufficiency         | Sequence    | API-call sequence covers all instruction requirements                                  |
| Minimality          | Sequence    | No unnecessary or redundant API calls                                                  |

An instance is "overall correct" only if all six criteria pass.

### Automated Scoring

ChatGPT-based assessment. Key design: direct yes/no prompts underperform task-based reformulations.

- **Specificity**: extraction task — ChatGPT extracts required parameter values, marks `#missing` for gaps. Score = 1 if all extracted.
- **Coherence**: next-sentence prediction — split instruction into sentences, check if each pair is logically connected.
- **Parameter Alignment**: two-step extract-then-compare — ChatGPT extracts parameters from instruction, then compares to ground-truth API call parameters.
- **Solvability, Sufficiency, Minimality**: direct assessment (these worked well without reformulation).

Validation against 50 human-annotated instances per dataset: overall accuracy 0.76–0.86, F1 0.81–0.92.

### In-Context Evaluation (ICE)

Core idea: evaluate a training instance by how well it works as a one-shot example. If showing this instance to a weak model helps it solve other tool-use tasks, it's probably good training data.

**Setup** (fixed, same for every evaluation):
- 10 hand-crafted API docs (get_weather, search_flights, etc.)
- 7 hand-crafted test queries with gold API-call sequences

**Procedure** — for each training instance x in the dataset:
1. Build a prompt: system instructions + 10 API docs + instance x as one-shot example + 7 test queries
2. Run a weak model (LLaMA-7B for ToolBench, Vicuna-7B for ToolAlpaca)
3. Compare model outputs to gold sequences via Levenshtein similarity
4. Average across 7 queries → ICE score for instance x

The only thing that changes between evaluations is which training instance is plugged in as the one-shot example. The 10 APIs and 7 test queries stay fixed. Repeat for all instances in the dataset.

**Theoretical motivation**: Dai et al. 2023 show that in-context learning implicitly performs something like gradient descent — so a good in-context example should share properties with good fine-tuning data.

**Score distributions**: bimodal on ToolAlpaca (clean dataset — most instances are either good or bad teachers), left-skewed on ToolBench (noisy dataset — most instances are poor teachers).

## Experiments

### Dataset Quality Assessment

| Criterion | ToolBench error % | ToolAlpaca error % |
|---|---|---|
| Specificity | 20.4% | 17.5% |
| Coherence | 22.1% | 4.1% |
| Solvability | 18.2% | 12.7% |
| Parameter Alignment | 47.9% | 33.1% |
| Sufficiency | 33.6% | 13.6% |
| Minimality | 45.1% | 15.9% |
| **Overall (any error)** | **84.0%** | **44.8%** |

Parameter Alignment is the worst dimension in both datasets — >33% of instances teach the model to hallucinate parameter values.

### Training Experiments

ToolBench: LLaMA-7B + LoRA, lr=5e-5, batch=2, ctx=4096, 2 epochs, 8×A10G.
ToolAlpaca: Vicuna-7B + LoRA, lr=2e-5, batch=2, 3 epochs, 4×A10G.

Test set: 420 manually validated ToolBench instances (37.7% of original test discarded as low-quality). ToolAlpaca: original 100-instance test with simulated tools.

| Subset | Size | ToolBench pass rate | ToolAlpaca pass rate |
|---|---|---|---|
| Random | 10K / 2K | 0.35 | 0.48 |
| Low ICE | 10K / 2K | 0.24 | 0.48 |
| High ICE | 10K / 2K | 0.43 | 0.54 |
| High Instruction | 10K / 2K | 0.49 | 0.52 |
| High Instr + Seq | 10K / 2K | 0.52 | 0.54 |
| High Instr + Seq + ICE | 10K / 2K | **0.54** | **0.55** |
| Original (full) | 73K / 4.2K | 0.45 | 0.56 |

Key results:
- Intrinsic criteria are more effective filters than ICE alone.
- Combining both gives marginal additional gain.
- 10K filtered > 73K unfiltered on ToolBench (0.54 vs 0.45).
- Effect weaker on ToolAlpaca because it's already cleaner.

### Scaling

Training set sizes 1K–20K on ToolBench. Performance increases consistently with more high-quality data, plateaus around 15–20K as the filtered pool is exhausted.

## Key Results

1. **84% of ToolBench is broken.** The most common failure is Parameter Alignment (48%) — the model is trained to hallucinate parameter values in nearly half of instances.
2. **14% of the data matches the full dataset's performance.** Filtering to ~10K clean instances from 73K matches or beats the unfiltered baseline.
3. **Intrinsic criteria > ICE for filtering.** Human-defined correctness dimensions are more reliable filters than the ICE proxy, but ICE adds marginal value on top.
4. **Task-based prompt reformulation** beats direct yes/no assessment for Specificity (+20% accuracy), Coherence (+8%), and Parameter Alignment (+4%).
5. **Test sets also need cleaning.** 37.7% of the original ToolBench test set was too low-quality to evaluate against.

## Core Concepts

- [[concepts/task-specific-quality-decomposition]] — the generalizable pattern: decompose quality along task structure (instruction-side vs output-side)
- [[concepts/data-filtering-paradigms]] — instance-level quality filtering as post-hoc curation
- [[concepts/verification-signals]] — six-criterion framework as a verification taxonomy for tool-use data
- [[concepts/llm-as-judge-methodology]] — ChatGPT as automated annotator with task-based reformulation
- [[concepts/critic-validation]] — automated quality assessment validated against human annotations
- [[concepts/multi-property-data-curation]] — multi-dimensional quality scoring (cf. Propella's property annotation)

## Relevance To Poolside

1. **Quality filtering as a cheap alternative to better generation.** The paper's closing claim — "if investing in better methods of data generation is costly, automatic post-hoc filtration can be a great alternative" — is directly relevant to Poolside's synthetic data pipelines. Any tool-use training data generated at scale should have quality gates.

2. **Parameter Alignment as the critical failure mode.** Nearly half of ToolBench teaches parameter hallucination. For Poolside's function-calling / agent data, parameter extraction fidelity should be a primary quality signal, not just task-level pass/fail.

3. **Multi-dimensional quality decomposition.** The six-criterion framework is a useful template for designing quality checks on any structured synthetic data (not just tool-use). Instruction-side checks (specificity, coherence, solvability) vs. output-side checks (alignment, sufficiency, minimality) is a clean decomposition.

4. **ICE as a cheap proxy.** The in-context evaluation idea — measuring educational value by how well an instance works as a one-shot example — could generalize to other data types. Worth exploring for code data where execution-based filtering is expensive.

5. **Less-is-more evidence.** Adds to the body of evidence ([[notes/papers/2023-lima|LIMA]], [[notes/papers/2024-phi-4-technical-report|Phi]], etc.) that small high-quality > large noisy for SFT. Poolside should consider quality-gated training set sizes rather than maximizing volume.

## Blaz Notes

Core insight (task-specific quality decomposition + less-is-more) is sound and useful. But specific numbers (84% error rate, 0.54 vs 0.45) should be treated as directionally correct rather than precise, given the methodological issues below.

**ChatGPT circularity.** ToolBench data generated by ChatGPT → quality assessed by ChatGPT → downstream pass rate judged by ChatGPT. The pipeline never escapes one model family's biases. Cannot distinguish "ChatGPT filters for what ChatGPT judges as good" from "filtering actually improves data quality." Cross-family evaluation (e.g., Claude judge, Llama evaluator) would be much more convincing.

**No diversity analysis under aggressive filtering.** Filtering to 14% of data and celebrating pass-rate improvement without checking what was lost. Could wipe out entire API categories, rare instruction types, or multi-step planning patterns. Quality-diversity tradeoff is the elephant in the room — acknowledged in limitations but not measured.

**Binary conjunction amplifies false rejections.** Six criteria each at ~75-85% accuracy via all-pass conjunction. Even a 10% false positive rate per criterion compounds to ~47% false rejection across 6 independent criteria. Almost certainly discarding good data along with bad. No analysis of this.

**50-instance validation set.** Confidence intervals on accuracy/F1 are wide. The 0.76 accuracy for Parameter Alignment could easily be 0.65-0.87. The whole filtering pipeline depends on these metrics being reliable.

**Only 7B + LoRA.** Quality sensitivity may differ at larger scale or full fine-tuning — larger models may be more or less robust to noisy training data.

**No comparison to simpler baselines.** Never asks whether the six-criterion decomposition outperforms a single "rate this training example 1-10" prompt, or even simpler heuristics (instruction length + API count). The ablation compares subsets but not methodology complexity.

**ICE is underpowered.** 7 hand-crafted test queries is an extremely small evaluation surface. May explain why ICE underperforms intrinsic criteria — not because the idea is wrong, but because the test set is too small to measure anything reliably.



## Key Follow-Ups / Jumping-Off Points

1. Does the six-criterion framework transfer to non-tool-use structured generation (e.g., code generation with test cases, multi-step reasoning)?
2. ICE uses LLaMA-7B / Vicuna-7B — would ICE scores from a stronger model be more predictive? Or is the weak-model signal actually more informative (because the instance needs to be clearer)?
3. The paper leaves dataset-level quality (diversity) as future work. How does diversity interact with instance quality when filtering aggressively?
4. [[notes/papers/2024-toolbench|ToolBench]] uses real-world APIs, [[notes/papers/2023-toolalpaca|ToolAlpaca]] uses synthesized APIs — the quality profile differs dramatically. What does this imply for Poolside's choice between real and synthetic tool definitions?

## Related Notes

- Papers:
  - [[notes/papers/2024-toolbench]] — primary benchmark (16K real APIs, 125K instances, 84% error rate)
  - [[notes/papers/2023-toolalpaca]] — secondary benchmark (2.3K synthetic APIs, 4.2K instances, 45% error rate)
  - [[notes/papers/2023-lima]] — foundational less-is-more result for SFT
  - [[notes/papers/2024-phi-4-technical-report]] — quality-focused synthetic data for pretraining
  - [[notes/papers/2024-qurating-selecting-high-quality-data-for-training-language-models]] — quality-based data selection for pretraining
  - [[notes/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale]] — multi-property annotation at scale
  - [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]] — survey of synthetic data quality metrics
  - [[notes/papers/2024-scaling-laws-for-data-filtering]] — scaling laws for filtered data
- Concepts:
  - [[concepts/data-filtering-paradigms]]
  - [[concepts/verification-signals]]
  - [[concepts/llm-as-judge-methodology]]
  - [[concepts/critic-validation]]
  - [[concepts/multi-property-data-curation]]
- Evals:
  - [[evals/bfcl-v3]], [[evals/bfcl-v4]] — related function-calling benchmarks

## Caveats

- **ChatGPT circularity**: data generated by ChatGPT, quality assessed by ChatGPT, pass rate judged by ChatGPT. No cross-family validation.
- **No diversity analysis**: filtering to 14% without measuring coverage loss. Quality-diversity tradeoff unmeasured.
- **Conjunction amplifies false rejections**: 6 criteria × ~75-85% accuracy each → substantial compounded false rejection rate unanalysed.
- **50-instance validation set**: wide confidence intervals on accuracy/F1; pipeline reliability rests on thin evidence.
- **7B + LoRA only**: quality sensitivity at larger scale / full fine-tuning unknown.
- **No simpler baseline comparison**: six-criterion framework never compared to single quality score or heuristic filters.
- **ICE underpowered**: 7 hand-crafted test queries; too small to reliably measure educational value.
- **Test set cleaned too**: 37.7% of ToolBench test discarded — potential selection bias favouring filtered training data.
- **Instance-level only**: dataset-level properties (diversity, coverage, difficulty distribution) explicitly out of scope.
