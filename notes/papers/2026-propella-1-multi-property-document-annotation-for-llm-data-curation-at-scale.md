---
arxiv: '2602.12414'
authors:
- Maximilian Idahl
- Benedikt Droste
- Bjorn Pluster
- Jan Philipp Harries
created: 2026-04-22
kind: paper
links:
  code: https://github.com/ellamind/inference-hive
  paper: https://arxiv.org/abs/2602.12414
  raw: '[[raw/papers/md/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale]]'
  source: https://arxiv.org/abs/2602.12414
owner: blaz
raw_pdf: raw/papers/pdf/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale.pdf
read: true
slug: propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale
tags:
- type/paper
- status/verified
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/data-mix
- domain/evals
- domain/models
- domain/synth-data
title: 'propella-1: Multi-Property Document Annotation for LLM Data Curation at Scale'
type: note
updated: '2026-05-10'
year: 2026
---

# propella-1: Multi-Property Document Annotation for LLM Data Curation at Scale

## Citation

- URL: https://arxiv.org/abs/2602.12414
- HTML: https://arxiv.org/html/2602.12414v1
- PDF: https://arxiv.org/pdf/2602.12414v1
- Authors: Maximilian Idahl, Benedikt Droste, Bjorn Pluster, Jan Philipp Harries
- Year / venue: 2026-02 arXiv preprint
- Raw PDF: [[raw/papers/pdf/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale.pdf]]
- Raw HTML: [[raw/papers/md/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale.html]]
- Figures: [[raw/images/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale/manifest]]

## Core Claim

**Single scalar document-quality scores are too lossy for LLM pretraining data curation.** Particularly the problem is that high reasoning depth but low pedagogical structure (e.g. legal analysis, technical specifications) may receive low educational scores (if you're using Fine-Web-Edu style pipeline). The paper claims that small specialized multilingual LLMs can annotate documents across 18 structured properties at scale, achieving higher agreement with frontier models than much larger general-purpose baselines, and enabling composable, interpretable filtering.

## Method

### Annotation Schema
18 properties across six categories. Output format: compact JSON without whitespace, schema-constrained via llguidance.

| Category            | Property                 | Type         | Values     |
| ------------------- | ------------------------ | ------------ | ---------- |
| Core Content        | Content Integrity        | Ordinal      | 4          |
| Core Content        | Content Ratio            | Ordinal      | 5          |
| Core Content        | Content Length           | Ordinal      | 4          |
| Classification      | One-Sentence Description | Free-text    | open vocab |
| Classification      | Content Type             | Multi-select | 18         |
| Classification      | Business Sector          | Multi-select | 37         |
| Classification      | Technical Content        | Multi-select | 7          |
| Quality & Value     | Content Quality          | Ordinal      | 5          |
| Quality & Value     | Information Density      | Ordinal      | 5          |
| Quality & Value     | Educational Value        | Ordinal      | 5          |
| Quality & Value     | Reasoning Indicators     | Ordinal      | 5          |
| Audience & Purpose  | Audience Level           | Ordinal      | 6          |
| Audience & Purpose  | Commercial Bias          | Ordinal      | 5          |
| Audience & Purpose  | Time-Sensitivity         | Ordinal      | 4          |
| Safety & Compliance | Content Safety           | Ordinal      | 5          |
| Safety & Compliance | PII Presence             | Binary       | 2          |
| Geographic          | Regional Relevance       | Multi-select | 14         |
| Geographic          | Country Relevance        | Multi-select | open vocab |

The rubric is an ~8,000-word document developed over two weeks. It defines permissible values, concrete positive/negative examples, decision points, and percentage-based thresholds for ordinal properties. It includes language-specific guidance for agglutinative languages, character encoding, and formality assessment.

### Model Family

Three decoder-only models at 0.6B, 1.7B, 4B parameters, based on Qwen-3. Context window: 64K. Fine-tuned with fp8 mixed-precision on 4× H100 GPUs in hours per variant. Inference system prompt ≈800 tokens (vs. 14K token rubric prompt used during label generation).

### Training Labels

A diverse document sample was annotated by multiple frontier LLMs prompted with the full rubric and strict schema. Training set: 57 languages, ~35% English, remainder spanning European, Arabic, CJK, Thai. Content types: web crawls, PDFs, curated datasets, code, math, post-training data. A small subset manually annotated for API-filter refusals.

### Inference Infrastructure

- **Serving**: SGLang + llguidance (schema-constrained generation, no post-processing needed).
- **Throughput (propella-1-4b)**: 27.0 docs/sec on 1× H100 fp8; ~10.3 GPU-hours per million documents. Throughput is prefill-dominated.
- **Distributed orchestration**: inference-hive (open-source SLURM-based). Example deployment: 500M FineWeb-2 documents in 3.5 hours on 3,936 A100 GPUs.

### Evaluation Setup

Test set: 3,000 documents. Reference: Gemini-3-Pro at "high" reasoning effort. Metrics: [[metrics/quadratic-weighted-kappa|QWK]] for 11 ordinal properties, [[metrics/f1-score|F1]] for binary PII Presence, [[metrics/intersection-over-union|IoU]] for 5 multi-select properties. Overall score is a weighted average; One-Sentence Description (free-text) excluded. Baselines use the full 14K rubric prompt; propella-1 uses its compact 800-token prompt.

## Key Results

### Annotator Agreement (overall weighted score)

| Model | Score | Notes |
|---|---|---|
| **propella-1-4b** | **0.779** | 4B specialized; compact prompt |
| Gemini-3-Flash-Preview | 0.778 | frontier, full rubric prompt |
| Gemini-2.5-Pro | 0.776 | frontier, full rubric prompt |
| propella-1-1.7b | 0.737 | 1.7B specialized |
| propella-1-0.6b | 0.729 | 0.6B specialized |
| Gemini-2.5-Flash | 0.700 | frontier |
| Mistral-Small-3.2-24B | 0.665 | general-purpose open |
| Gemma-3-27B | 0.604 | general-purpose open |
| Qwen3-30B | 0.556 | general-purpose open |
| Qwen3-4B | 0.454 | general-purpose open |
| Gemma-3-4B | 0.295 | general-purpose open |
| SmolLM3-3B | 0.227 | general-purpose open |

fp8 vs. bf16: negligible quality difference confirmed.

### propella-annotations Dataset

3,005,080,817 annotations by propella-1-4b across: FineWeb-2 (~1.63B), FinePDFs, HPLT 3.0, Nemotron-CC, SYNTH, finewiki, German Commons. Covers English + 14 European languages. Released CC-BY-4.0 on HuggingFace.

### Dataset Case Studies

**German multi-source profiling** (FineWeb-2 vs. FinePDFs vs. HPLT 3.0 vs. German Commons):
- FinePDFs: 21.4% "excellent" content quality vs. 2.4% in FineWeb-2; ~12× more analytical reasoning; 7.4% high educational value vs. 0.6%.
- HPLT 3.0: higher rates of content fragments and degraded documents than FineWeb-2.

**Nemotron-CC quality-tier audit** (10K docs/tier):
- Nemotron-CC "high" tier still contains meaningful rates of heavy commercial bias, thin information density, and incomplete content integrity.
- Single-score tiers do not cleanly separate individual property dimensions.

**Cross-language variation in FineWeb-2** (6 European languages):
- Substantial differences in content quality, commercial bias, educational value, and content type distributions across languages.
- Commercial bias and information density show the largest cross-language variance.
- Implication: uniform quality thresholds across languages are suboptimal; language-specific filtering is warranted.

## Critique

*Our synthesis — distinct from the paper's claims.*

- **No downstream training evaluation.** The paper explicitly acknowledges it has not yet shown that filtering with propella annotations improves trained model quality. The entire value proposition rests on this undemonstrated link. Agreement with Gemini-3-Pro is a proxy; it is not the goal.
- **Shared frontier bias.** Annotators are trained on frontier-LLM labels; evaluation reference is also a frontier LLM. High agreement is expected and may reflect shared systematic errors rather than correctness. Human ground truth is absent.
- **Rubric is a design choice.** The 18 properties and their values reflect this team's curation philosophy. Properties relevant to Poolside code data — e.g., correctness, test coverage, API novelty, algorithmic depth — are not in the schema. Adoption requires schema re-design, not just plug-and-play.
- **Multi-select properties are harder to aggregate.** Content Type (18 values) and Business Sector (37 values) produce complex joint distributions that are difficult to threshold compositionally. The paper does not address how to operationalise multi-select signals in a data-mixture recipe.
- **The "no post-processing" claim depends on constrained decoding.** The throughput advantage vs. general LLMs is partly because SGLang + llguidance enforces schema. This is an engineering constraint, not a model capability — any sufficiently small model could be served the same way.

## Blaz Notes

### Key takeaway

The propella-1 recipe generalises beyond this specific schema: annotate a sample with a frontier LLM using a well-defined rubric and strict JSON schema, fine-tune a small model on those labels, serve with constrained decoding. The result is a cheap, fast, schema-correct classifier that matches frontier agreement at a fraction of inference cost. Any domain (code quality, reasoning depth, safety) with a clear rubric and enumerated output values is a candidate for this treatment. Note that the compact 800-token inference prompt is a distilled summary of the rubric — the model has internalised the rubric at training time, not inference time.

### Traditional filtering landscape

- **Heuristic filtering** (length, char ratios, repetition removal). *Works:* fast, cheap, language-agnostic in principle. *Doesn't:* FineWeb's own ablation shows many heuristics have low or negative impact; Su et al. (2025) find disabling heuristics for high-quality documents improves accuracy and that heuristics remove 18% of high-quality tokens. Language-specific assumptions break for languages without spaces or sentence-final punctuation.
- **Perplexity filtering** (CCNet — KenLM on Wikipedia). *Works:* captures fluency, works across languages, cheap at scale. *Doesn't:* biased toward Wikipedia-like prose; cannot discern semantic quality; scores fluent-but-nonsensical text highly; penalizes code, conversational text, and technical writing.
- **Classifier-based quality scoring** (FineWeb-Edu paradigm — LLM labels → distilled classifier). *Works:* scalable, stronger signal than heuristics, has driven concrete improvements. *Doesn't:* single scalar conflates multiple quality dimensions; English-centric; introduces topic bias — filtering for educational value systematically favors education/history/science and removes entertainment/business/travel (FineWeb-Edu's own analysis).
- **Register-based annotation** (HPLT 3.0 — hierarchical text-type labels). *Works:* Luukko et al. (2025) show register-based selection affects LLM performance; text type is a signal independent of quality. *Doesn't:* register ≠ quality; doesn't surface safety, commercial bias, information density, or reasoning depth.
- **Multi-dimensional scalar scoring** (QuRating, DataMan, Meta-rater, RedPajama V2 — multiple scalar axes). *Works:* multi-dimensional selection can improve convergence speed vs. single score. *Doesn't:* still produces scalars per dimension without clean enumerated rubrics; mostly English; no joint composability at schema level.

### Framework design principles

Three design constraints the authors applied when choosing and defining properties:

1. **Each property enables a concrete filtering use case.** A property that doesn't drive a decision at curation time was dropped. The iterative rubric process started with a larger set and removed properties that failed this test.
2. **Properties should be complementary, not redundant.** Overlap between properties reduces interpretability and makes compositional filtering ill-defined. The schema was tested for this.
3. **Annotation values should use enumerated categories with clear rubric definitions.** Open-ended or continuous values introduce inter-annotator inconsistency at scale. Enumerated + rubric = reproducible at model-distillation time.

### Per-property concrete curation scenarios

From the paper directly (§3.2):

- **Educational Value** → curriculum-style training schedules; weight high-educational-value data more early in training.
- **Reasoning Indicators** → select content rich in analytical and explanatory reasoning; directly relevant for training reasoning models.
- **Commercial Bias** → filter out marketing-dominated content that may degrade model objectivity.
- **Time-Sensitivity** → distinguish evergreen reference material from ephemeral news; freshness-aware data selection or versioned corpora.
- **Content Type + Business Sector** → domain-targeted dataset construction; e.g. build a finance-focused or legal-focused corpus slice.
- **Content Safety + PII Presence** → safety filtering pipeline; PII removal for compliance.
- **Content Integrity + Content Ratio** → remove broken, boilerplate, or navigation-heavy pages where main content is a small fraction of the document.
- **Information Density** → filter thin content (e.g. one-sentence pages, placeholder articles) that adds tokens without signal.
- **Audience Level** → match content complexity to training objectives; avoid over-indexing on highly technical or highly simplified text depending on target use.
- **Regional / Country Relevance** → geo-targeted corpus construction; balance regional representation in multilingual training.

## Open Threads

- Does any propella property predict downstream model training improvement? This is the unanswered question the authors themselves flag.
- What is a code-specific property schema for Poolside? The web-document schema is likely a useful starting point, not the final design. High-value candidates: algorithmic complexity, test coverage presence, API / library novelty, code correctness (runnable/not), style consistency.
- Could propella-style annotation replace or complement near-duplicate filtering for code seeds? The Content Integrity and Content Ratio properties are adjacent to repetition signals.
- Does a property-aware annotator replace the need for separate classifiers (e.g., a separate code-quality classifier) or compose with them?
- inference-hive for Poolside data pipelines? The SLURM orchestration is directly compatible with typical cluster setups.
- Language-specific filtering thresholds: the cross-language variance finding directly applies to any multilingual Poolside pretraining data.

## Related Notes

- Concepts:
  - [[concepts/multi-property-data-curation]]
  - [[concepts/regmix]]
  - [[concepts/data-repetition]]
- Hypotheses:
  - [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
- Experiments:
  - [[experiments/dat-440-raw-code-rephrasing]]
- Papers:
  - [[notes/papers/2024-nemotron-cc-transforming-common-crawl-into-a-refined-long-horizon-pretraining-dataset]]
  - [[notes/papers/2025-olmo-3]]
- Figures:
  - ![[raw/images/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale/figure-01.png]]
  - ![[raw/images/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale/figure-03.png]]
  - ![[raw/images/papers/2026-propella-1-multi-property-document-annotation-for-llm-data-curation-at-scale/figure-04.png]]
