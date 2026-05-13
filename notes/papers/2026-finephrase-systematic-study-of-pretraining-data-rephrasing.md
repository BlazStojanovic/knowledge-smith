---
aliases:
- 2026-finephrase
- finephrase
arxiv: '2604.13977'
authors:
- Joel Niklaus
- Atsuki Yamaguchi
- Michal Stefanik
- Guilherme Penedo
- Hynek Kydlicek
- Elie Bakouch
- et al. (12 authors)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.13977
  raw: '[[raw/papers/md/2026-finephrase-systematic-study-of-pretraining-data-rephrasing]]'
  source: https://arxiv.org/abs/2604.13977
owner: blaz
raw_pdf: raw/papers/pdf/2026-finephrase-systematic-study-of-pretraining-data-rephrasing.pdf
read: true
slug: finephrase-systematic-study-of-pretraining-data-rephrasing
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
- domain/models
- stage/pretrain
title: 'FinePhrase: Systematic Study of Prompt, Generator, Source for Pretraining-Data
  Rephrasing'
type: note
updated: '2026-05-10'
year: 2026
---

# FinePhrase: Systematic Study of Prompt, Generator, Source for Pretraining-Data Rephrasing

- **Paper title**: How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data
- **arXiv**: [2604.13977](https://arxiv.org/abs/2604.13977)
- **Authors**: Joel Niklaus, Atsuki Yamaguchi, Michal Stefanik, Guilherme Penedo, Hynek Kydlicek, Elie Bakouch, et al. (12 authors)
- **Year / venue**: 2026-04 (arXiv preprint)
- **Raw**: [[raw/papers/pdf/2026-finephrase-systematic-study-of-pretraining-data-rephrasing.pdf]]
- **Grounding axis**: [[maps/grounding/real-code-anchor]] (Axis 2, generalised to real-document anchor). Prompt-conditioned rephrasing of web text.
- **Output shape**: (source document, rephrased document) pairs in four structured formats — math word problem, FAQ, table, tutorial. Released dataset: **FinePhrase**, 486 B tokens / 1.35 B samples.
- **Filter / verification**: No explicit per-sample filter beyond generator-prompt adherence. Quality judged through downstream macro-average.
- **Training stage**: Pretrain (continued pretrain or from-scratch mixin).

## Method

Prompt-conditioned rephrasing at scale. Three axes ablated:

1. **Prompt design.** Compares established rephrasing prompts (Nemotron-CC Diverse-QA, Extract-Knowledge, Distill, Wikipedia, Knowledge-List; REWIRE Guided-Rewrite; BeyondWeb Continue / Summarize) against four new **structured output** formats: math word problem, FAQ, table, tutorial.
2. **Generator model.** 135 M → 27 B across Gemma 3 and SmolLM2 families.
3. **Source data.** DCLM, FineWeb-HQ (score 4-5), FineWeb-LQ (score 0-1), Cosmopedia.

Downstream (pre-train ablation): 
* 1.2 B Qwen 2 architecture, 
* trained from scratch on **21 B tokens** (64 H100s, seq 4096, batch 512). 
* Macro-average over 12 benchmarks — see [[#Evaluations]] below. 
	* Prompting: [[concepts/cloze-evaluation|3-shot cloze format]] throughout.

## Key results

### Rephrasing strategy ablation

Macro-average × 100 at 1.2 B / 21 B tokens. All synthetic variants mixed 50/50 with DCLM.

| Strategy | Source | Score | Δ DCLM |
|---|---|---|---|
| DCLM (baseline) | — | **13.77** | — |
| Diverse QA Pairs | Nemotron-CC | 14.58 | +0.81 |
| Continue | BeyondWeb | 13.73 | −0.04 |
| Guided Rewrite | REWIRE | 13.72 | −0.05 |
| Distill | Nemotron-CC | 13.17 | −0.60 |
| Wikipedia | Nemotron-CC | 13.14 | −0.63 |
| Knowledge List | Nemotron-CC | 13.11 | −0.66 |
| Summarize | BeyondWeb | 13.01 | −0.76 |
| Extract Knowledge | Nemotron-CC | 11.81 | −1.96 |
| **Tutorial** | this paper | **14.30** | **+0.53** |
| **FAQ** | this paper | **14.45** | **+0.68** |
| **Table** | this paper | **14.83** | **+1.06** |
| **Math** | this paper | **15.31** | **+1.54** |
| FinePhrase-Table (full scale) | this paper | **17.18** | **+3.41** |

### Generator-size ablation
* Gemma 3: ***no benefit above 1B parameters.** Gemma 3 1B (15.31) > Gemma 3 27B
* SmoLM2: ***1.7B (15.88) > 360M (14.45) > 135M (12.69)***

There is a note that for more complicated modes a better model does in fact improve performance (REWIRE)

When comparing across model families, you should look for good performance on tasks which are similar to what we're trying to do (this is just common sense).

### Mix-in and source data ablation

**Synthetic-only vs. 50/50 mixed** (Math generator, DCLM as mix-in):

| Format | Synth-only | Mixed 50/50 | Δ mix |
|---|---|---|---|
| Math | 15.20 | 15.31 | +0.11 |
| Table | 13.74 | 14.83 | +1.09 |
| FAQ | 13.12 | 14.45 | +1.33 |
| Tutorial | 12.32 | 14.30 | +1.98 |

Math is nearly self-sufficient synthetic; the other formats depend heavily on mix-in to preserve commonsense/NLU coverage.

**Source quality × mix-in quality** (Table format; all rows use a 50/50 mix):

| 𝒟_source | 𝒟_mix | Score |
|---|---|---|
| FineWeb-LQ | FineWeb-LQ | 9.63 |
| Cosmopedia | Cosmopedia | 10.36 |
| DCLM | DCLM | 13.69 |
| FineWeb-HQ | FineWeb-HQ | 14.30 |
| FineWeb-LQ | FineWeb-HQ | 12.99 |
| Cosmopedia | FineWeb-HQ | 13.88 |
| FineWeb-HQ | DCLM | 14.32 |
| **DCLM** | **FineWeb-HQ** | **14.77** |

Key reads: upgrading the mix-in from LQ→HQ gains +3.36 on a LQ source (9.63→12.99); upgrading the source from LQ→HQ with a fixed HQ mix-in gains only +1.31 (12.99→14.30). DCLM rephrased + FineWeb-HQ mix-in (14.77) marginally outperforms FineWeb-HQ rephrased + FineWeb-HQ mix-in (14.30) — mix-in quality dominates source quality.

## Critique

*Our synthesis — distinct from the paper's claims.*

**Raised by Blaz:**

- **All transforms are simple rephrasing.** Math / FAQ / table / tutorial are four prompt templates applied to existing documents. The "structured" framing describes output shape, not input coverage. No new knowledge is synthesised — only restructured. Compare to [[notes/papers/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale]] which adds a second organic grounding (real-user-query templates) to produce genuinely new (instruction, answer) pairs.
- **Only tested at small scale.** Downstream evaluation is a single 1.2 B model × 21 B tokens. No scaling curves. Claims about format ordering (table > math > FAQ > tutorial) at this scale may not hold at 7 B / 70 B parameters or 500 B–1 T tokens.

**Additional observations:**

- **"No benefit beyond 1B generator" is easy to over-read.** The macro-average is benchmark-bounded; it may simply be insensitive to the extra knowledge a larger generator would inject. Rephrasing is a low-content-bandwidth task — a smaller model suffices because we're not asking for synthesis, only reformatting. The claim generalises poorly outside this operation class.
- **Mix-in dependence.** "Pure synthetic underperformed across all prompts" — the headline gains require a 50 / 50 mix with original web. Synthetic is a *lens* on the source, not a substrate. Frontier-scale pretraining cannot rely on this ratio indefinitely.
- **FinePhrase dataset released but not yet pretrain-validated.** 486 B tokens is a substantial release, but no frontier-scale model has been fully pretrained on it to corroborate the 1.2 B / 21 B extrapolation.
- **Source-data ablation is the paper's quiet contribution.** The paper's direct claim: *"the inherent quality of 𝒟_source becomes less dominant when paired with robust mix-in data."* In other words, the quality of the seed documents fed into the rephrasing pipeline matters less if 𝒟_mix is curated — the mix-in quality is the dominant lever. Quantitatively: the source quality gap (FineWeb-HQ vs FineWeb-LQ) shrinks from **4.67 points** when D_mix = D_source (14.30 vs 9.63) to **1.31 points** when D_mix = FineWeb-HQ (14.30 vs 12.99). Upgrading the mix-in from LQ→HQ on a fixed LQ source gains +3.36 (9.63→12.99); upgrading the source from LQ→HQ with a fixed HQ mix-in gains only +1.31. Practical implication: invest in mix-in curation before source curation.

## Notes

- Cluster: WRAP [2401.16380](https://arxiv.org/abs/2401.16380), Nemotron-CC Synth, Cosmopedia, [[notes/papers/2025-recycling-the-web-a-method-to-enhance-pre-training-data-quality-and-quantity-for-language-models|REWIRE]] [2506.04689](https://arxiv.org/abs/2506.04689), BeyondWeb, [[notes/papers/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale]]. All are document-anchored Axis 2 / Axis 3 rephrasing recipes. FinePhrase is the cleanest public *ablation* across prompt × generator × source.
- Structured-output prompts (table, FAQ) align with an evaluation pattern already visible in synthetic-data papers (Cosmopedia, phi synthesis): pedagogical structure as a diversity / quality proxy. This paper quantifies it.
- The 30× cost reduction vs. REWIRE is largely **generator-size** (1.7 B vs larger REWIRE generator) plus speculative decoding — not prompt-specific.
- Released dataset: FinePhrase on HuggingFace (verify release state before citing a URL).

## Evaluations

12 benchmarks, macro-averaged. [[concepts/cloze-evaluation|3-shot cloze format]] throughout. Grouped by the paper's own 6 categories.

### General Knowledge

| Benchmark      | What it measures                                                        | Note                                                         |
| -------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| **ARC-Easy**   | Science MCQ, elementary level                                           | [[evals/arc]]                                                |
| **MMLU Redux** | Cleaned subset of MMLU; 3,000 re-annotated questions across 30 subjects | [[evals/mmlu]] (full MMLU note; Redux is a filtered version) |

### Reading Comprehension

| Benchmark | What it measures | Note |
|---|---|---|
| **SQuAD v2** | Extractive QA + unanswerable questions on Wikipedia passages | no eval note |
| **DROP** | Discrete reasoning over paragraphs (arithmetic, counting, sorting) | [[evals/drop]] |

### Reasoning

| Benchmark | What it measures | Note |
|---|---|---|
| **OpenBookQA** | Elementary science QA requiring multi-hop reasoning over a small fact base | no eval note |
| **XCSQA** | Cross-lingual CommonsenseQA; English subset used here | no eval note |

### NLU

| Benchmark | What it measures | Note |
|---|---|---|
| **WinoGrande** | Coreference / commonsense pronoun resolution at scale | [[evals/winogrande]] |
| **PIQA** | Physical intuition QA | [[evals/piqa]] |
| **HellaSwag** | Sentence completion requiring commonsense / world knowledge | [[evals/hellaswag]] |

### Math

| Benchmark | What it measures | Note |
|---|---|---|
| **GSM8K** | Grade-school math word problems, multi-step arithmetic | [[evals/gsm8k-v2]] |

### Table QA

| Benchmark | What it measures | Note |
|---|---|---|
| **WikiTableQuestions** | Complex QA over Wikipedia HTML tables | no eval note |
| **TriviaQA** | Trivia QA with evidence documents; tests factual retrieval | [[evals/triviaqa]] |

### Key per-task pattern

Structured formats (table, math) sharply improve reading comprehension (SQuAD v2 +18.72 for table) and factual knowledge (ARC +7.92 for FAQ). They hurt commonsense NLU: HellaSwag −7.66 for table vs DCLM. Original web data remains load-bearing for PIQA / HellaSwag / WinoGrande. Mix-in is what preserves those scores.

## Prompt Templates

All templates use `[TEXT]` as the source document placeholder. Source: [github.com/huggingface/finephrase/prompts](https://github.com/huggingface/finephrase/tree/main/prompts).

### Structured (new — this paper)

**Math** (`format/math.md`)
```
Rewrite the document to create a mathematical word problem based on the numerical data or relationships in the text. Provide a step-by-step solution that shows the calculation process clearly. Create a problem that requires multi-step reasoning and basic arithmetic operations. It should include the question followed by a detailed solution showing each calculation step. Output only the problem and solution, nothing else.

Document:
[TEXT]
```

**FAQ** (`format/faq.md`)
```
Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions a reader would have about this topic, then provide clear, direct answers. Order questions logically—from foundational to advanced, or by topic area. Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing else.

Document:
[TEXT]
```

**Table** (`format/table.md`)
```
Rewrite the document as a structured table that organizes the key information, then generate one question-answer pair based on the table. First extract the main data points and organize them into a clear table format with appropriate headers using markdown table syntax with proper alignment. After the table, generate one insightful question that can be answered using the table data. Provide a clear, concise answer to the question based on the information in the table. Output only the table followed by the question-answer pair, nothing else.

Document:
[TEXT]
```

**Tutorial** (`format/tutorial.md`)
```
Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet points where appropriate to enhance clarity. Preserve all essential information while ensuring the style feels didactic and easy to follow. Output only the tutorial, nothing else.

Document:
[TEXT]
```

### Nemotron-CC (`nemotron/`)

**Diverse QA Pairs** — ask up to 8 questions covering yes/no, open-ended, multi-choice, comparison, reading comprehension, and problem-solving types. Each Q&A tagged `Question:` / `Answer:` on separate lines. Plain text only.

**Extract Knowledge** — rewrite as textbook/Wikipedia-style passage; focus on humanities, sciences, tech, law, etc.; retain examples and reasoning; do not add or alter details; plain text.

**Distill** — condense to accurate, informative paraphrase; preserve key concepts, technical terms, examples, and reasoning; do not add new claims; plain text.

**Wikipedia-style Rephrasing** — one-shot paraphrase in high-quality Wikipedia English; output prefixed with `Here is a paraphrased version:`.

**Knowledge List** — extract factual information as a concise, organised bullet list; information-dense; no titles or headings.

### REWIRE (`rewire/`)

**Guided Rewrite** (`guided_rewrite_improved.md`) — long chain-of-thought meta-reasoning prompt: identify the core problem, break it down, plan a strategy, then rewrite the original draft from the author's perspective with substantially better formatting, coherence, and structure. Output must be the rewritten content only — no commentary or meta-text.

### BeyondWeb (`beyondweb/`)

**Continue** — continue the text in the same style; start the continuation directly.

**Summarize** — standalone summary; do not reference the source text; start directly; no preamble.

## Related notes

- [[concepts/rephrasal-operations]]

## Open threads

- Do the 1.2 B / 21 B format rankings survive at frontier scale? This is the core extrapolation question for the paper.
- How does FinePhrase compose with [[notes/papers/2026-fineinstructions-scaling-synthetic-instructions-to-pre-training-scale]]-style dual-grounded synthesis? Both are Axis-2-adjacent and complementary.
- Is there a format-space ceiling at four prompts, or does adding more (e.g., debate, worked-example, dialog) continue to help?
- The no-benefit-beyond-1B claim invites the inverse test: at what scale of student-model does it start to break?
