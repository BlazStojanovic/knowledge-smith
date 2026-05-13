---
aliases:
- 2026-fineinstructions
- fineinstructions
arxiv: '2601.22146'
authors:
- Ajay Patel (UPenn)
- Colin Raffel (Toronto / Vector)
- Chris Callison-Burch (UPenn)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.22146
  raw: '[[raw/papers/md/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale]]'
  source: https://arxiv.org/abs/2601.22146
owner: blaz
raw_pdf: raw/papers/pdf/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale.pdf
read: false
slug: fineinstructions-scaling-synthetic-instructions-to-pre-training-scale
tags:
- type/paper
- source/primary
- status/verified
- confidential/public-source
- domain/general
- domain/llm
- domain/synth-data
- domain/pretraining
- domain/data-mix
- domain/evals
- domain/models
- stage/pretrain
- stage/sft
title: 'FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale'
type: note
updated: '2026-05-10'
year: 2026
---

# FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale

## Citation

- **arXiv**: [2601.22146](https://arxiv.org/abs/2601.22146)
- **Authors**: Ajay Patel (UPenn), Colin Raffel (Toronto / Vector), Chris Callison-Burch (UPenn)
- **Year / venue**: 2026-01-29 arXiv preprint; ICML keyword tag in metadata
- **Raw PDF**: [[raw/papers/pdf/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale.pdf]]
- **Raw HTML**: [[raw/papers/md/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale.html]]
- **Dataset**: [huggingface.co/fineinstructions](https://huggingface.co/fineinstructions) (1B+ pairs)

## Core Claim

A pipeline that templatises ~18M real user queries and instantiates them against pretraining documents produces billions of (instruction, answer) pairs that — when used as the *sole* objective for from-scratch pretraining — outperform standard next-token pretraining and prior synthetic-pretraining baselines (IPT, WRAP, Nemotron-CC) at matched tokens on free-form response benchmarks.

## Methodology

![[raw/images/papers/2026-fineinstructions/figure-02-pipeline.png]]

Four-stage pipeline (paper §3). Distilled task-specific models do the heavy lifting; Llama-3.3 70B Instruct generates the silver-standard data used to train them via DataDreamer.

1. **Template extraction (§3.1).** Mine ~18M user-written queries from WildChat (657K), LMSys Chat (559K), Reddit QA (~7.47M), GooAQ (~3.01M), Dolly, OAsst1, NoRobots, HelpSteer, plus prompt libraries (Anthropic, LangChain, Awesome-ChatGPT-Prompts) and academic templates (FLAN, P3, NaturalInstructions). Filter with OpenAI Moderation API; decontaminate via the Tulu 3 procedure. Convert each query to a generic template by replacing entity / scenario spans with `<fi>short description</fi>` tags. Distil this into a 1B-parameter "Query Genericizer" (fine-tuned Llama-3.2 1B Instruct) trained on ~50K silver templates.
2. **Document → template matching (§3.2).** A two-stage fine-tune of BGE-M3 (FAISS index over template "compatible-document descriptions"). Stage 1: standard cosine-similarity contrastive fine-tune on hard positives/negatives produced by an LLM compatibility judge. Stage 2: introduces a custom **Gaussian pooling layer** that produces *K+1* embeddings per document (1 global mean + *K=5* Gaussian-weighted local embeddings, σ=0.05, α=1.0, centred at evenly-spaced fractions of the sequence). Stage 2 trains for chunk-localised representations — labels include which chunk index *k* is relevant, and loss is taken on that chunk's local embedding. The end result: in the final dataset, **Pearson 0.99** between the retrieving chunk index and the location of the answer excerpt within the source document; most excerpts come from 19–71% through the document.
3. **Instantiation + answer generation (§3.3).** A distilled 3B-parameter "Instantiator" (fine-tuned Llama-3.2 3B Instruct) instantiates each template against its matched document and extracts excerpt(s) as the answer, with light rephrasing allowed. Hard constraint: **excerpt-text ratio ≥ 0.80** in the final answer — the generator acts mainly as a *retrieval-and-reshape* model. Decode-time efficiency trick: long verbatim spans are emitted as `<excerpt>start<...>end</excerpt>` tags and expanded programmatically post-decoding, so the model only generates short excerpt anchors. Trained in two rounds; ~5% of examples are kept where the LLM judged the template *incompatible* with the document (so the model can learn to output `null` instead of forcing a low-quality generation).
4. **Judge + filter (§3.4).** Off-the-shelf Flow Judge (3.8B distilled judge) scores each (instruction, answer) on a 5-point Likert scale; pairs scoring ≥ 4 are retained.

**Retrieval (§3.2).** Cosine threshold of **0.865** on the fine-tuned embeddings. Among matches, weighted random sampling preserves the `<fi>`-tag-count distribution observed in real-world LLM queries, so simple and complex templates remain proportionally represented.

**Pretraining (§4.2).** Lingua framework on 8×H100; Llama-3 tokenizer. 1.8B-parameter models trained for one epoch on Nemotron-CC datasets (~300B tokens) and four epochs on IPT-derived datasets (~23B tokens). Six retrieved templates per document → six (instruction, answer) pairs per document; randomly kept until total token count matches the source document's token count (≈3 pairs/document on average). Token budget rolls over between documents. Format: `Instruction: {{instruction}}\n\nAnswer: {{answer}}` (token count includes formatting). Baselines pre-trained in their native formats. Token-for-token controlled comparison.

## Experiments & Results

![[raw/images/papers/2026-fineinstructions/figure-01-efficiency-scatter.png]]

**Headline (1.8B, Table 1).** All models trained on the same number of tokens.

| Method | MixEval Std | MixEval Hard | MT-Bench-101 | AlpacaEval (FI win %) |
|---|---|---|---|---|
| *IPT corpus, ~23B* | | | | |
| Standard pretrain | 17.8 | 14.0 | 1.9 | FI **73.6 %** (Δ 47.2) |
| IPT | 19.8 | 16.7 | 2.4 | FI **68.2 %** (Δ 36.4) |
| **FineInstructions** | **31.7** | **19.2** | **2.8** | — |
| *Nemotron-CC corpus, ~300B* | | | | |
| Standard pretrain | 24.0 | 17.1 | 3.5 | FI **63.6 %** (Δ 27.2) |
| WRAP | 22.8 | 18.4 | 3.6 | FI **65.1 %** (Δ 30.2) |
| Nemotron-CC Q&A only | 27.1 | 18.9 | 3.4 | FI **76.1 %** (Δ 52.2) |
| Nemotron-CC mixture | 24.5 | 16.7 | 3.6 | FI **65.9 %** (Δ 31.8) |
| **FineInstructions** | **33.0** | **21.8** | **3.9** | — |

Paper reports **~69%** relative improvement on MixEval-Std vs. standard pretrain on the IPT corpus, and **~39%** on Nemotron-CC. AlpacaEval head-to-head: FineInstructions wins against every comparator.

**Scale (Appendix F, Table 5).** Models at 300M / 1.8B / 7B; FineInstructions at fixed size is competitive with or beats baselines trained at the *next* size up under matched compute and tokens.

**Judge ablation (Appendix E, Table 4).** Filter-on vs. filter-off mostly improves AlpacaEval win rates (e.g., on Nemotron-CC Q&A, 68.7 % → 76.1 %). MixEval / MT-Bench-101 improvements smaller and not always monotone (one Nemotron-CC FI MT-Bench-101 cell is *higher without judging*: 20.2 vs. 19.2).

**Diversity (§6.1, Appendix D).**

- 4.3M unique templates instantiated from Nemotron-CC base; **no single template > 0.09 %** of generated instructions.
- Template-utilisation power fit: `y = 16,891 · x^0.24` with `r² = 0.96` (templates used vs. documents processed).
- Source mix of generated instructions: GooAQ ~50 %, Reddit QA ~27 %, LMSys Chat ~9 %, WildChat ~6 %, all others ≤ 1 % each.
- Categorical mix: Science **36.6 %**, Personal Life 14.7 %, Reasoning Task 11.0 %, Medicine 10.4 %, Tasky-vs-QA 6.4 %, **Math 0.58 %**, **Code 0.25 %** (zero-shot Llama-3.3 70B classification, non-mutually exclusive).

![[raw/images/papers/2026-fineinstructions/figure-03a-all-sunburst.png]]

## Core Concepts

- **Grounding type:** **(M) metadata grounding** on the question side × **Axis 3 non-code artifact (real document)** on the answer side → **dual-side organic grounding**. Both labels are recorded in [[maps/grounding/landscape#3. Cross-cutting observations|the grounding landscape, §3 cross-cutting observations]]. Most prior Axis-2 / Axis-3 work grounds only one side. *We infer* this is the paper's main conceptual contribution over WRAP (answer-side rephrasing only) and Self-Instruct / Code Alpaca (neither side organic).
- **Pretrain-stage instruction objective.** Stage positioning is unusual — pretrain rather than SFT — closer to WRAP / Nemotron-CC / Phi-style "synthetic pretraining" than to Self-Instruct / Evol-Instruct. Authors argue the instruction-tuning objective during pretraining closes the train/test mismatch with downstream usage.
- **Grounded reshaping.** The ≥ 0.80 excerpt-ratio constraint is a hard-coded hallucination-prevention control. Makes the generator primarily a retrieval-plus-reshape model rather than a free writer. Worth naming as a distinct generator role in [[concepts/rephrasal-operations]].
- **Template scale as a diversity lever.** 18M organic-scraped templates vs. hundreds / thousands in prior work. Templates are organic-scraped (WildChat, LMSys Chat, Reddit), not synthetic — a cheap diversity source that doesn't require a generator.
- **Existing concepts:** [[concepts/rephrasal-operations]], [[maps/grounding/real-code-anchor]] (real-document anchor, generalised beyond code), [[maps/grounding/non-code-artifacts]] (real user-query templates) — **dual-grounded**.
- **Concepts to extract:** *grounded reshaping* as a distinct generator role; *dual-side organic grounding* as a recipe pattern.

## Blaz Notes

-

## Key Follow-Ups

- **Code-domain port.** Paper's instructions are 0.25 % code by mix. Obvious port for Poolside: source user-written *coding* queries (StackOverflow, LeetCode discussion, GitHub issues, internal pair-programming logs) as templates and match them against code documents. Question: does Gaussian-chunk pooling work on code, where semantic chunking by token-fraction is less natural than for prose?
- **Excerpt-ratio binding at scale.** Is the ≥ 0.80 constraint binding at all model scales, or does it hurt at 7B+ where generator capability exceeds document quality? Appendix F shows FI still wins at 7B but doesn't ablate the constraint.
- **Organic-template coverage vs. synthetic taxonomies.** How does the organic template distribution compare to synthetic taxonomies ([[maps/grounding/knowledge-structure]] — InsTag, Explore-Instruct) for coverage of long-tail tasks? The 4.3M-templates / 1.08B-instructions ratio (Pareto-thin) is a useful baseline.
- **Mid-train, not from scratch.** Paper trains from scratch. Most realistic Poolside use is mid-training or as an SFT mix. Open question: what fraction of FI-shaped data in a mid-train corpus retains the AlpacaEval/MT-Bench-101 lift?
- **Filter-stage interaction with answer style.** Judge ablation suggests filter helps AlpacaEval much more than MixEval. *We infer* the Flow-Judge rubric is rewarding response-style features that line up with AlpacaEval's judge — worth checking the judge-judge correlation before relying on the lift.

## Caveats

- **No log-prob benchmarks.** Authors deliberately exclude log-prob multiple-choice benchmarks because pre-training on instruction-answer pairs assigns very low probability to short-form answers. Leaves a gap on classic pretrain capability eval (HellaSwag-style) — the comparison is not just "FI vs. pretrain" but "FI on free-form benchmarks vs. pretrain on free-form benchmarks", which is the regime where pretrain is weakest.
- **MT-Bench-101 spread is small.** The paper itself notes MT-Bench-101 is reference-free and yields low spread (1.9–3.9 across all methods at 1.8B). Most of the differentiation comes from MixEval and AlpacaEval; the headline "wins on three benchmarks" reads stronger than the absolute numbers warrant on MT-Bench-101.
- **Judge ablation is non-monotone.** Table 4: on Nemotron-CC, FI without judging beats FI with judging on MT-Bench-101 (20.2 vs. 19.2). Filter is not a free win across all metrics.
- **Safety / decontamination relies on opaque external services.** OpenAI Moderation API for harm filtering and Tulu 3 procedure for decontamination — both reproducible only with effort. *Worth flagging* if Poolside replicates this pipeline.
- **Eval judges are GPT-5 mini and GPT-4-Turbo.** Strong-but-asymmetric: the same judge family is used across all comparators, but absolute scores are not portable to other judges.
- **Single seed, single model arch implied.** No variance reporting in the headline tables; replication risk on the 0–2 % spreads between baselines on MT-Bench-101.
- **Math 0.58 %, Code 0.25 % by category.** *Inference*: as-shipped, the dataset is heavily skewed toward general-knowledge prose. A code-focused training run on as-published FineInstructions would be data-starved on code instructions.
