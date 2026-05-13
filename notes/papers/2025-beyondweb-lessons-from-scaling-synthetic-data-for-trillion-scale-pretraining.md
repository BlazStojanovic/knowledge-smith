---
arxiv: '2508.10975'
authors:
- DatologyAI
- ':'
- Pratyush Maini
- Vineeth Dorna
- Parth Doshi
- Aldo Carranza
- Fan Pan
- Jack Urbanek
- Paul Burstein
- Alex Fang
- Alvin Deng
- Amro Abbas
- Brett Larsen
- Cody Blakeney
- Charvi Bannur
- Christina Baek
- Darren Teh
- David Schwab
- Haakon Mongstad
- Haoli Yin
- Josh Wills
- Kaleigh Mentzer
- Luke Merrick
- Ricardo Monti
- Rishabh Adiga
- Siddharth Joshi
- Spandan Das
- Zhengping Wang
- Bogdan Gaza
- Ari Morcos
- Matthew Leavitt
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2508.10975
  raw: '[[raw/papers/md/2025-beyondweb-lessons-from-scaling-synthetic-data-for-trillion-scale-pretraining]]'
  source: https://arxiv.org/abs/2508.10975
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-beyondweb-lessons-from-scaling-synthetic-data-for-trillion-scale-pretraining.md
raw_pdf: raw/papers/pdf/2025-beyondweb-lessons-from-scaling-synthetic-data-for-trillion-scale-pretraining.pdf
read: true
slug: beyondweb-lessons-from-scaling-synthetic-data-for-trillion-scale-pretraining
tags:
- type/paper
- source/primary
- status/draft
- domain/synth-data
- domain/pretraining
title: 'BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining'
type: note
updated: '2026-05-11'
year: 2025
---

# BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining

## Citation

- URL: https://arxiv.org/abs/2508.10975
- PDF: https://arxiv.org/pdf/2508.10975
- Authors: DatologyAI Team — Pratyush Maini, Vineeth Dorna, Parth Doshi, Aldo Carranza, Fan Pan, Jack Urbanek, Paul Burstein, Alex Fang, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Charvi Bannur, Christina Baek, Darren Teh, David Schwab, Haakon Mongstad, Haoli Yin, Josh Wills, Kaleigh Mentzer, Luke Merrick, Ricardo Monti, Rishabh Adiga, Siddharth Joshi, Spandan Das, Zhengping Wang, Bogdan Gaza, Ari Morcos, Matthew Leavitt
- Affiliation: DatologyAI
- Year / venue: arXiv preprint, August 2025
- Raw PDF: [[raw/papers/pdf/2025-beyondweb]]
- arXiv HTML: not available (`No HTML for '2508.10975'`)

## Core Claim

A "no-silver-bullet" lessons-paper for **source-rephrasing** synthetic pretraining data at trillion-token scale. Introduces **BeyondWeb**, a rephrasing pipeline that combines high-quality web seeds, distributional style matching (especially conversational format), and explicit diversity across rephrasing strategies. On the headline 8B / 180B-token comparison it beats RedPajama by +7.1pp and Nemotron-Synth by +2.6pp on a 14-benchmark average; a 3B model trained on BeyondWeb outperforms most 8B baselines at the same token budget; speedups: 7.7× over RPJ, 2.7× over Nemotron-Synth, 5.5× over Cosmopedia, 2.0× over QA-WRAP to reach the same baseline accuracy.

The paper's contribution is not the single number — it's the seven-RQ ablation that disentangles *why* synthetic data helps and *what* breaks at scale.

## Key Paper Ideas

The paper's framing introduces the **generator-driven vs source-rephrasing** taxonomy and then runs seven RQs to characterise the source-rephrasing side.

- **Two paradigms of synthetic pretraining data** (§2):
  - *Generator-driven* (TinyStories, phi, Cosmopedia): large LLM generates training corpora *de novo* from topic / persona / role prompts. Bottlenecked by generator cost + biases + collapse risk.
  - *Source-rephrasing* (WRAP, REWIRE, Nemotron-CC, Phi-4, BeyondWeb): smaller LLM rephrases existing web documents into higher-quality / task-aligned formats. Lower compute, broader coverage, less collapse-prone.
- **Distillation hypothesis is incomplete** (RQ1, §4.2). A naive *summarisation* prompt over web data closes most of the gap to Cosmopedia (46.7% vs 47.1%) — so much of generator-driven gains is just per-token information density. BeyondWeb still beats summarisation by +3.7pp, so density is *one* mechanism, not the only one.
- **Data wall is breachable, but not by naive synthesis** (RQ2, §4.3). With a 20B-token data budget split into "first half / second half / continuation": 2× repeat = 45.5%, naive 8B-LLM continuation = 46.2%, full-data upper bound = 46.2%, **BeyondWeb = 50.4% (+4.2pp over upper bound)**. Strategic synthetic surpasses the natural-data ceiling; naive synthetic does not.
- **Rephrase high-quality, not low-quality, sources** (RQ3, §4.4). HQ-Synth (rephrased HQ web) > LQ-Synth (rephrased LQ web) > LQ-Web baseline. But HQ-input alone is not sufficient — BeyondWeb still beats HQ-Synth by +1.2pp because *strategy* matters too.
- **Distributional style matching matters but saturates** (RQ4, §4.5). Conversational text is only 3.67% of web, yet chat is the dominant inference use case. Upsampling to 50% conversational lifts 5-shot accuracy from 43.2% → 44.1% (+0.9pp); gains saturate beyond ~20% conversational share.
- **Diversity is what sustains gains at trillion-token scale** (RQ5, §4.6). Single-style strategies (Cosmopedia textbooks, QA-WRAP single-format) plateau and even overfit at 1B / ~50× Chinchilla; multi-strategy BeyondWeb continues improving across 1B / 3B / 8B all the way to 180B tokens. This is the strongest lesson in the paper.
- **Rephraser family is largely interchangeable** (RQ6, §4.7). Across OLMo-2-7B, Phi-4-14B, Mistral-7B-v0.3, Llama-3.1-8B as rephrasers, downstream accuracy spans 48.9–49.9% (<1pp). Notably, **rephraser benchmark accuracy does not predict synthetic-data quality**: OLMo-2-7B has the lowest general accuracy (59.6%) but produces the highest-quality synthetic data (49.9%).
- **Rephraser size saturates around 3B** (RQ7, §4.8). Llama-3.2-1B → 3B gives +1.5pp; 3B → 8B gives only +0.4pp. Production rephrasing does not need a frontier model.

## Methodology

- **Setup** (§3, §4.1). Llama-3.2-1B / 3B and Llama-3.1-8B trained on a 20B mixture (60% RPJ + 40% synthetic for headline runs; 50/50 HQ-Web + synthetic for the RQ ablations). Source corpus is a high-quality DCLM subset filtered with the DatologyAI curation stack. Llama-3.1-8B-Instruct is the default rephraser.
- **Baselines.** RedPajama (RPJ), Cosmopedia v2 (Mixtral-8×7B textbooks), WRAP (QA-format paraphrase), Nemotron-Synth (high-quality subset of Nemotron-CC).
- **Eval.** lighteval over 14 benchmarks (8 from FineWeb + 6 more): ARC-C/E, BoolQ, COPA, CommonsenseQA, HellaSwag, MMLU, OpenbookQA, PIQA, RACE-H/M, SIQA, SciQ, WinoGrande. Reports averages of 0-shot + 5-shot using cloze-form (CF) scoring (probabilities restricted to valid choices, à la HuggingFace OpenLLM leaderboard / OLMES).
- **Headline runs.** 1B trained for 1T tokens; 3B + 8B trained for 180B tokens. 3 sizes × 5 datasets.
- **Ablation harness.** Default ablation: 1B Llama-3.2 trained on 20B tokens, 50/50 HQ-Web / synthetic. The 10B HQ-Web pool is *the same tokens* shared across baselines and rephrased — so the source knowledge is held fixed.
- **Diversity / repetition probe (§4.3).** Documents are split at the paragraph midpoint; the second half is replaced with either (i) repeat-of-first-half, (ii) Llama-3.1-8B continuation of first-half, or (iii) BeyondWeb rephrasing. The continuation arm uses the *second* half deliberately so the rephraser hasn't seen the target during its own pretrain.

## Experiments

- Headline 14-benchmark mean across 1B/3B/8B × {RPJ, QA-WRAP, Cosmopedia, Nemotron-Synth, BeyondWeb} (Table 1, Fig 1).
- RQ1: Cosmopedia vs simple summarisation prompt vs BeyondWeb (Fig 2).
- RQ2: 20B controlled corpora (Fig 3 — split / repeat / continuation diagram; Fig 4 — accuracy curves).
- RQ3: HQ Synth + HQ Web (Repeat) vs LQ Synth + HQ Web vs LQ Web + HQ Web vs BeyondWeb (Fig 5).
- RQ4: conversational ratio sweep at 10 / 20 / 50 % vs RPJ baseline (Fig 6).
- RQ5: Δ-accuracy-vs-RPJ training curves at 1B / 3B / 8B (Fig 7).
- RQ6: rephraser-family swap — OLMo-2-7B / Phi-4-14B / Mistral-7B-v0.3 / Llama-3.1-8B (Fig 8). Right panel of Fig 8 is the load-bearing observation: rephraser-LLM benchmark accuracy ⊥ synthetic-data quality.
- RQ7: rephraser size 1B / 3B / 8B (Fig 9).

## Key Results

- **Headline.** 8B / 180B BeyondWeb = **63.7%** (vs RPJ 56.6, QA-WRAP 58.4, Cosmopedia 58.6, Nemotron-Synth 61.1). 3B BeyondWeb = 60.8% (>= all-but-one 8B baseline at the same tokens). 1B / 1T BeyondWeb = 57.4%. Consistent 13/14 task wins at 1B; 12/14 at 3B; 13/14 at 8B (Table 1, Fig 1).
- **Speedup to baseline accuracy.** 7.7× vs RPJ, 5.5× vs Cosmopedia, 2.7× vs Nemotron-Synth, 2.0× vs QA-WRAP (Fig 1 right).
- **RQ1 — distillation.** Summarisation (46.7%) ≈ Cosmopedia (47.1%) ≫ RPJ-HQ (45.5%); BeyondWeb (50.4%) is +3.7pp above summarisation, so synthesis is more than just compression.
- **RQ2 — data wall.** Strategic synthetic (BeyondWeb 50.4%) > full-data upper bound (46.2%) > naive continuation (46.2%) > 2× repeat (45.5%). Naive synthetic ≤ no-synthetic; strategic synthetic breaks through.
- **RQ3 — quality vs novelty.** HQ Synth + HQ Web (49.2%) > LQ Synth + HQ Web (48.6%) > LQ Web + HQ Web (45.6%). BeyondWeb at 50.4% adds another +1.2pp on top.
- **RQ4 — style matching.** Saturates at ~20% conversational share; +0.9pp ceiling.
- **RQ5 — diversity at scale.** At 1B / ~50× Chinchilla, baselines (Cosmopedia, QA-WRAP) saturate or degrade; BeyondWeb keeps improving (Fig 7). The single most important practical lesson.
- **RQ6 — rephraser invariance.** OLMo-2-7B (59.6% general accuracy) → 49.9% synth quality; Phi-4-14B (66.6%) → 49.5%. Rephraser quality and rephrased-data quality are decorrelated — the rephrasing task is "constrained transformation" not "knowledge generation".
- **RQ7 — rephraser size.** 1B → 3B = +1.5pp; 3B → 8B = +0.4pp. 3B is the practical sweet spot.

## Core Concepts

- Existing concepts touched:
  - [[concepts/content-provenance-axis]] — BeyondWeb sits at **Tier 1 (pure seed-bound)**, paired with WRAP and FinePhrase. Updated in this pass.
  - [[concepts/rephrasal-operations]] — BeyondWeb is the trillion-token-scale empirical study of **archetype #2 (format imposition)** with explicit *strategy diversification* on top.
  - [[concepts/diversity]] — RQ5 is the strongest empirical evidence in the KB for "single-strategy synthetic saturates at scale; multi-strategy sustains". Lit-pointer added in this pass.
  - [[concepts/data-repetition]] — RQ2 quantifies the gap between naive repetition (-0.7pp), naive continuation (~0pp), and strategic synthesis (+4.2pp over the full-data upper bound). Update block added.
  - [[concepts/synthetic-data-formalism]] — clean instance of $G$ as a transformation $\mathcal{S} \to \mathcal{S}$ with $\alpha \approx 1$ in the formalism's information-flow decomposition.
- Concepts to extract: nothing new is load-bearing enough to warrant a fresh page on a light pass. Candidate for later: a **"rephraser-invariance"** concept if RQ6 replicates in code / math domains.

## Relevance To Poolside

*Our interpretation, explicitly labelled.*

- **Validates the source-rephrasing direction over generator-driven for pretrain-scale synthetic data.** Confirms WRAP and Nemotron-Synth's design choices and provides the strongest published evidence that a 3B rephraser is enough — directly relevant to any Poolside rephrasing pipeline that picks rephraser size by gut feel.
- **Diversity-at-scale finding is the most important transferable lesson.** RQ5's training-dynamic plot — single-strategy saturation at 50× Chinchilla compute — applies wherever we are scaling a single-format rephrasing pipeline (e.g. raw-code rephrasing). *We infer:* the equivalent risk for code is "rephrase-as-textbook" or "rephrase-as-QA" pipelines saturating well before our trillion-scale code budgets are exhausted; mitigation is to run several heterogeneous rephrasing strategies in parallel rather than scaling a single best-format.
- **Rephraser invariance (RQ6) suggests our rephraser model choice is mostly a cost question.** Worth replicating before relying on this for code: code rephrasing may have a different generator-quality dependency than web text. Concrete follow-up: re-run RQ6 with a small set of code-instruct rephrasers on a code subset.
- **The summarisation result (RQ1) is a cheap baseline we should probably run.** A trivial summarisation pass closes most of Cosmopedia's gap; this is a strong "what does it cost to do nothing fancy?" probe.
- **Their training-dynamic plotting protocol (Fig 7 — running mean across checkpoints, Δ-vs-baseline y-axis) is a clean visualisation idiom worth reusing** for our own long-horizon ablations.

## Blaz Notes
- A bit opaque paper, ablations looks sound - but they are likely witholding info 
- Two modes -> summary and continuation, careful design is needed to scale
- Only pretraining, downstream post-training effects are not obvious
- A big part of effect is style alignment (distributional style matching)

## Key Follow-Ups / Jumping-Off Points

- **RQ6 in the code domain.** Does generator-LLM benchmark performance decorrelate from rephrased-code quality the same way it does for web text? Cross-link to [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]] if relevant.
- **What is "diversity" operationally in BeyondWeb?** The paper says "diverse generation strategies" but does not enumerate the BeyondWeb prompt set. The diversity operationalisation is opaque from the paper alone — see Caveats.
- **Cross-domain test of style matching (RQ4).** Code-prediction at inference is closer to "code-completion in IDE" than to "chat" — the analogous "natural deployment style" benchmark for code is not obviously conversational. What is the right style for a code-pretraining mix?
- **Composition with non-rephrasing axes.** All RQs hold the source corpus fixed at HQ-DCLM / RPJ. Open: does the diversity finding interact with execution-grounded ([[maps/grounding/execution-traces]]) or repo-anchored ([[maps/grounding/repository-multifile]]) signals when those exist?

## Related Notes

- Concepts: [[concepts/content-provenance-axis]], [[concepts/rephrasal-operations]], [[concepts/diversity]], [[concepts/data-repetition]], [[concepts/synthetic-data-formalism]]
- Sibling papers: [[notes/papers/2024-rephrasing-the-web-a-recipe-for-compute-and-data-efficient-language-modeling]] (WRAP — same first author, BeyondWeb's direct predecessor), [[notes/papers/2025-recycling-the-web-a-method-to-enhance-pre-training-data-quality-and-quantity-for-language-models]] (REWIRE — concurrent rephrase-medium-quality-into-high), [[notes/papers/2024-cosmopedia]] (generator-driven counter-example, the "Cosmopedia" baseline), [[notes/papers/2023-textbooks-are-all-you-need]] (phi-1 lineage), [[notes/papers/2023-tinystories]] (generator-driven origin), [[notes/papers/2025-scaling-laws-of-synthetic-data-for-language-models]]
- Maps: [[maps/grounding/landscape]] (rephrasing of web text is closest to Axis 2/3 seed-bound material on the code-grounding map; the map is code-oriented and the fit is rough)
- Hypotheses: —
- Experiments: —
- Figures (in PDF): Fig 1 (Pareto frontier + speedup), Fig 2 (RQ1 summarisation), Fig 3 (RQ2 corpus split), Fig 4 (RQ2 accuracy), Fig 5 (RQ3 quality combinations), Fig 6 (RQ4 style ratio), Fig 7 (RQ5 diversity scaling), Fig 8 (RQ6 rephraser family + decorrelation), Fig 9 (RQ7 rephraser size)

## Caveats

- **"Diversity" is under-specified in the paper.** RQ5 is the most important finding but the paper does not enumerate the full BeyondWeb prompt set or the strategy-mixing distribution. The reader cannot reproduce the diversity recipe — only the headline numbers. The Datology curation platform is described as "to be detailed in a follow-up release".
- **Not a Pareto-optimal study of generator-driven approaches.** Cosmopedia is the only generator-driven baseline; Phi-1 / TinyStories themselves are not retrained at the same scale. The "rephrasing wins" conclusion is *vs Cosmopedia*, not vs every generator-driven recipe.
- **20B-token RQ ablations on 1B Llama generalise loosely to 8B / 180B.** The headline is at 8B, but the seven RQs run at 1B / 20B. The authors verify some findings hold at 8B, but most ablations are not re-run at scale — typical for cost reasons but worth flagging for any quantitative reuse.
- **Eval is academic / multiple-choice-heavy.** The 14-benchmark suite is dominated by knowledge-MC and commonsense-MC (MMLU, HellaSwag, ARC, BoolQ, …). Strong performance does not directly imply strong code, math, or long-form reasoning. The paper does not run code-specific or math-specific evals.
- **Cloze-form scoring choice.** Restricting probabilities to valid choices ([[concepts/cloze-evaluation]]) inflates absolute accuracy vs free-form scoring — comparison across papers using different scoring is not 1:1.
- **Naive-continuation upper-bound caveat (paper's own).** The continuation rephraser (Llama-3.1-8B) has seen the 20B source corpus during its own pretrain, so the continuation arm may be borrowing parametric knowledge rather than purely "filling in". Authors flag this in §4.3.
- **Single rephrasing-prompt suite tested.** RQ6 swaps the rephraser model with prompts held fixed; the dual ablation (fix rephraser, sweep prompts) is not reported.
- **Memorisation / contamination audit not reported.** No Rouge-overlap or near-duplicate scan against eval sets — see [[metrics/rouge-overlap]] for the canonical TinyStories-style audit. With BeyondWeb being whole-document rephrasing of the open web, contamination of academic eval prompts via the source web pool is a non-trivial risk.
- **License of BeyondWeb itself.** The data is not (as of writing) released; AFM4.5B is the customer-facing artefact. The paper is a methodology study, not an open-data release.
