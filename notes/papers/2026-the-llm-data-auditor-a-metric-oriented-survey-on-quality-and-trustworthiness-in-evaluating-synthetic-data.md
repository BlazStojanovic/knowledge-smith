---
arxiv: '2601.17717'
authors:
- Kaituo Zhang
- Mingzhi Hu
- Hoang Anh Duy Le
- Fariha Kabir Torsha
- Zhimeng Jiang
- Minh Khai Bui
- et al. (12 authors)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.17717
  raw: '[[raw/papers/md/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]]'
  source: https://arxiv.org/abs/2601.17717
owner: blaz
raw_pdf: raw/papers/pdf/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data.pdf
read: true
slug: the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data
tags:
- type/paper
- status/active
- source/primary
- confidential/public-source
- domain/synth-data
- domain/evals
- domain/code
- domain/reasoning
- domain/agents
title: 'The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness
  in Evaluating Synthetic Data'
type: note
updated: '2026-05-10'
year: 2026
---

# The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data

## Citation

- URL: https://arxiv.org/abs/2601.17717
- PDF: https://arxiv.org/pdf/2601.17717
- Authors: Kaituo Zhang, Mingzhi Hu, Hoang Anh Duy Le, Fariha Kabir Torsha, Zhimeng Jiang, Minh Khai Bui, et al. (12 authors)
- Year / venue: 2026-01-25 arXiv preprint (v2)
- arXiv: 2601.17717v2
- Categories: cs.AI, cs.LG
- Raw PDF: [[raw/papers/pdf/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data.pdf]]
- Raw HTML: [[raw/papers/md/2026-the-llm-data-auditor.html]]

## Core Claim

Most research on LLM-based synthetic data focuses on generation methods; rigorous *intrinsic* evaluation of the generated data is underdeveloped. The paper proposes the **LLM Data Auditor** framework — a unified metric taxonomy that organises evaluation of synthetic data across six modalities into two pillars: **Quality** (validity, fidelity, diversity, utility) and **Trustworthiness** (safety, faithfulness, privacy, robustness, fairness, provenance). By auditing representative methods against this taxonomy, the authors expose systematic evaluation practice gaps.

![[Pasted image 20260423170139.png]]
## Key Paper Ideas

- **Data-centric, not model-centric.** Shift from extrinsic evaluation (downstream task performance) to intrinsic evaluation (direct data quality measurement before it enters training).
- **Two-pillar taxonomy.** Quality = {validity, fidelity, diversity, utility}. Trustworthiness = {safety, faithfulness, privacy, robustness, fairness, provenance}. Not every dimension applies to every modality — the paper maps which apply where.
- **Six modalities.** Text, symbolic/logical reasoning, tabular, semi-structured (graph/JSON/log), vision-language, agent data. Each gets its own generation survey, quality metrics, trustworthiness metrics, evaluation gap analysis, and usage review.
- **Evaluation practice gaps.** Every modality section includes a table auditing representative methods against the taxonomy dimensions (✓/△/×). The consistent finding: most methods evaluate validity (final-answer correctness) but neglect faithfulness, diversity, safety, and robustness.
- **Process verifiability > surface mimicry.** As reasoning LLMs surpass average human performance, fidelity should shift from distributional similarity to human artifacts toward verifiable correctness of reasoning processes (Section 8.2).
- **Static metrics miss model collapse.** Current metrics give single-round snapshots; recursive train-generate loops need longitudinal trajectory monitoring to detect distributional drift before collapse (Section 8.1).
- **Trust-utility Pareto frontier.** Safety filtering, differential privacy noise, and alignment impose an "alignment tax" on diversity/utility. Future work should characterise the Pareto frontier and enable application-dependent trade-offs (Section 8.3).

## Methodology

This is a survey paper; the methodology is taxonomic rather than experimental:

1. **Modality-indexed literature review.** For each of six modalities, catalogue representative generation methods, then audit their published evaluations.
2. **Metric formalisation.** For each quality/trustworthiness dimension within each modality, define concrete metrics with mathematical formulations (118+ equations across the paper).
3. **Gap analysis tables.** For each modality, construct a table (Tables 2–7) rating representative methods as ✓ (explicitly evaluated), △ (partially covered), or × (not reported) per dimension.
4. **Cross-modal synthesis.** Identify recurring gaps across modalities (faithfulness, diversity, safety consistently under-evaluated) and formulate three open challenges.

## Key Results

### Quality Metrics — Per Modality

#### Text (Section 2.2) — metric notes created

| Dimension | Metrics                                                                                                                                                                      |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Validity  | [[metrics/grammatical-acceptability-rate]] (GAR), [[metrics/usl-h]] (USL-H, hierarchical dialogue validity)                                                                  |
| Fidelity  | [[metrics/embedding-distribution-similarity]] (EDS), [[metrics/instruction-following-fidelity]] (Acc^strict_prompt, IFEval strict), [[metrics/ruber]], [[metrics/pmi-faith]] |
| Diversity | [[metrics/self-cosine-similarity]] (Self-CosSim, semantic), [[metrics/type-token-ratio]] (TTR), [[metrics/distinct-n]], [[metrics/ngram-entropy]] (Ent-n)                    |

#### Symbolic / Logical Reasoning (Section 3.2) — metric notes created

| Dimension | Metrics |
| --------- | ------- |
| Validity  | [[metrics/verification-accuracy]] (Acc_verify, generic domain-checker), [[metrics/strict-proof-accuracy]] (Acc_proof, strict proof-graph match + FAIRR consistency), [[metrics/execution-success-rate]] (PassRate, unit-test execution — link to [[concepts/pass-at-k-methodology]]) |
| Fidelity  | [[metrics/self-consistency-accuracy]] (Acc_SC + Agree), [[metrics/judge-human-correlation]] (ρ_LLM-human + Q_reason rubric), [[metrics/reward-model-accuracy]] (Acc_RM) |

#### Tabular Data (Section 4)

> Skipped — not within KB scope. See Auditor §4 for statistical fidelity (column-distribution tests, correlation-matrix distance), privacy (DCR, MIA), and fairness (SPD, EO, EOp) metrics for tabular synthetic data.

#### Semi-structured Data (Section 5.2) — metric notes created

Three sub-modalities: **Graph**, **JSON**, **Log**. Semi-structured data sits between rigid relational schemas and unstructured text.

| Sub-modality | Dimension | Metrics |
|---|---|---|
| Graph | Validity | [[metrics/graph-validity-rate]] (Valid_rule — fraction passing task-specific rule checker) |
| Graph | Fidelity | MMD² on graph descriptors (degree distributions, clustering coefficients, motif counts); Fréchet ChemNet Distance (FCD) for molecular graphs |
| Graph | Diversity | Novelty (fraction not identical to prompt examples), Uniqueness (non-duplicate rate among valid graphs) |
| Graph | Utility | Accuracy / F1 of GNN trained on synthetic+real data vs real-only |
| JSON | Validity | [[metrics/schema-correctness-rate]] (CorrectnessRate = parsable ∧ schema-compliant; ValidJSONRate = parsable-only) |
| JSON | Fidelity | Mean Match Percentage (MMP — field overlap with ground-truth instances) |
| JSON | Utility | TaskAcc (exact-match accuracy of extracted answer vs reference) |
| Log | Validity | Parsing Accuracy (PA — template+variable exact match), Grouping Accuracy (GA), Template Accuracy F1 (FTA) |
| Log | Fidelity | Variable P/R/F1 (predicted vs true runtime variables), Log Level Accuracy (L-ACC), Average Ordinal Distance (AOD), BLEU/ROUGE on templates |
| Log | Utility | Δ_M (downstream metric change when augmenting real data with synthetic logs) |

**Generation methods surveyed (§5.1):**

- **Graph**: Training-free (LLM4GraphGen — direct prompting; Generate-on-Graph — KG evidence retrieval; ontology-grounded KG construction). Learning-based (GraphJudge — SFT discriminator; GAG — multi-agent scalable synthesis; GraphMaster — evaluation-driven iterative refinement).
- **JSON**: Constrained decoding (vLLM structured outputs, Outlines, LM Format Enforcer — token masking at inference time). Learning-based (SchemaBench — RL with schema-validator rewards; Think Inside the JSON — GRPO for smaller models).
- **Log**: LLMs predict logging attributes accurately but fail at surface-form fidelity (best BLEU = 0.249 on LogBench). Domain specialisation via post-training (AUCAD) bridges the gap.

#### Vision-Language Data (Section 6)

> Skipped — multimodal / vision-grounded training data is explicitly out of scope for this KB (see CLAUDE.md §Scope). See Auditor §6 for image/video generation metrics (FID, CLIPScore, IS), provenance (watermark robustness), and evaluation practice gaps.

#### Agent Data (Section 7)

Framed via world models as internal simulators that capture environment dynamics (Li et al. 2025, arXiv [2510.16732](https://arxiv.org/abs/2510.16732)). Data products classified by primary downstream use into three categories. See [[concepts/trajectory-synthesis]] §Agent-data product taxonomy and [[concepts/synthetic-data-formalism]] §Grounding extensions for formalism mapping.

**Three data-product categories (§7.1):**

| Category | What is generated | Representative methods |
|---|---|---|
| **Environment / task data** | World setups, task configs, PDDL problems, temporal-logic specs | L3M+P (PDDL from NL), [[notes/papers/2024-selp\|SELP]] (NL→LTL), [[notes/papers/2025-t3-planner\|T3 Planner]] (STL verification loop), [[notes/papers/2024-partnr\|PARTNR]] (multi-agent collaboration tasks), [[notes/papers/2025-ttsg\|TTSG]] (text-to-traffic-scene) |
| **Control / decision data** | Action traces, parametrisation plans, agent trajectories | LLM-driven simulation (closed-loop DT traces), [[notes/papers/2025-grid-agent\|Grid-Agent]] (power grid DT control), [[notes/papers/2022-saycan\|SayCan]] (language-grounded planning), [[notes/papers/2023-code-as-policies\|Code as Policies]], LLM Trainer (demonstration augmentation) |
| **Perception / telemetry data** | Sensor streams, rendered observations, defect descriptions | DefectTwin (multimodal inspection), SceneCraft (3D scene rendering), BlenderLLM (programmatic 3D assets) |

**Quality metrics (§7.2):**

| Dimension | Metrics | Detail |
|---|---|---|
| **Validity** | ExecRate, SR_valid, PC | ExecRate = fraction of actions that execute without error. SR_valid = fraction of episodes where all goal predicates are satisfied. PC = fraction of goal predicates achieved (partial completion). PARTNR uses proposition-based evaluation with temporal constraints. TTSG uses binary text-matching correctness. |
| **Fidelity** | FID, FVD, SSIM, LPIPS, SemAlign | Perception-channel metrics: FID/FVD compare real vs synthetic feature distributions (Gaussian fit); SSIM compares luminance/contrast/structure; LPIPS computes perceptual distance in deep feature space. **SemAlign** is a reference-free proxy: CLIP cosine similarity between text prompt and rendered outcome, applicable when no paired ground truth exists. Mean semantic fidelity = average SemAlign over prompt-outcome pairs. |
| **Diversity** | AD, RD, VBench | AD (agent diversity) = unique agents / total across repeated generations. RD (road diversity) = unique roads / total. Both from TTSG. VBench decomposes video generation into axis-wise scores (subject consistency, motion smoothness, prompt adherence). **Gap:** only TTSG explicitly reports diversity metrics. |
| **Utility** | SR_eval, SimSteps, Offloading, CR, $G_t$ | SR_eval = binary success rate over evaluation episodes. SimSteps = mean environment steps per episode (efficiency). Offloading = fraction of sub-tasks completed by the robot vs total (collaborative settings). CR = expected collision indicator (driving; lower = safer). $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$ (discounted return for RL policy evaluation). |

**Trustworthiness — safety metrics (§7.3):**

Agent-data trustworthiness evaluation is almost entirely safety-focused. Other dimensions (fairness, privacy, robustness, provenance) are undeveloped for this modality.

| Family               | Metrics                                                                                                                                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Rule violations**  | RVR = per-episode violation count / exposure (steps or distance). Per-type RVR$^{(t)}$ breaks down by infraction category (collision, red-light, lane invasion, wrong-way). Exposure-normalised (infractions per km) as in CARLA Leaderboard. |
| **Formal safety**    | SafetySat = $\frac{1}{N}\sum_i \mathbb{1}(\sigma^{(i)} \models \phi)$ where $\phi$ is an LTL/STL specification. SELP maps NL→LTL with equivalence voting; T3 Planner uses STL verification in the loop.                                       |
| **Hazard rejection** | Rejection rate = fraction of hazardous tasks refused. Risk rate = fraction executed. SafeAgentBench (arXiv [2412.13178](https://arxiv.org/abs/2412.13178)) reports low rejection and non-trivial risk for current embodied LLMs.              |
| **Proximity**        | TTC = $d(t)/(-\dot{d}(t))$ when closing; MDC = $\min_t \lVert p_\text{ego}(t) - p_\text{other}(t) \rVert$. Lower = higher collision risk.                                                                                                     |
| **Speed / comfort**  | MSCR = fraction of timesteps meeting minimum speed. Jerk_RMS = root-mean-square jerk (smoothness). HardBrakeRate = fraction of timesteps with longitudinal acceleration below braking threshold.                                              |
| **Route**            | RI = $1 - \text{completed distance} / \text{planned route length}$. Higher = worse (early termination).                                                                                                                                       |
| **Diagnosis**        | Violation diagnosis accuracy: P/R/F1 of automated detectors (including MLLM-based auditors) for identifying safety violations from logs/images. SeeUnsafe proposes Information Matching Score for structured response alignment.              |

**Evaluation practice gap (§7.4):**

| Method | Validity | Fidelity | Diversity | Utility | Safety |
|---|---|---|---|---|---|
| TTSG | ✓ | △ | ✓ | ✓ | ✓ |
| PARTNR | ✓ | × | × | ✓ | × |
| SELP | ✓ | × | × | ✓ | ✓ |
| Grid-Agent | ✓ | × | × | ✓ | △ |
| T3 Planner | ✓ | △ | × | ✓ | ✓ |

Three key gaps: (1) **Diversity** — only TTSG reports quantitative diversity; other methods cannot demonstrate mode/scenario coverage. (2) **Fidelity** — only consistency proxies (prompt-scene matching); no distributional comparison of synthetic vs reference trajectories. (3) **Diagnostic safety** — broad benchmark metrics (collisions, spec satisfaction) but no per-violation-type breakdown.

**Usages (§7.5):**

- **Environment/task data**: testbed construction for planning and safety verification (L3M+P persistent world-state graphs, SELP constrained plan generation, PARTNR collaborative evaluation suites). Trial-and-error traces (failed plans + verifier diagnostics + repairs) from self-correction loops are reusable for training.
- **Control/decision data**: supervision for imitation learning (SayCan, Code as Policies), reward design via LLM-authored reward functions ([[notes/papers/2024-eureka|Eureka]]), offline RL on language-conditioned static logs, AI-feedback scoring of archived trajectories. DT control logs can be relabeled and reused for policy refinement.
- **Perception/telemetry data**: evaluation signals via LLM-as-judge on multimodal inputs, vision-language reward learning from execution videos, 4D world modelling from temporally-aligned multi-domain observation corpora (OmniWorld).

### Trustworthiness Metrics — Key Definitions

| Dimension                    | Key metrics                                                                                                                                                                                         | Modalities where defined |                                                                                |                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------ | --------------- |
| Faithfulness                 | [[metrics/attribution-to-sources]] (umbrella: Attr_AIS, Attr_auto, Pres_intent, Pres_Lev, F1_AP)                                                                                                    | Text                     |                                                                                |                 |
| Faithfulness (reasoning)     | [[metrics/step-validity-rate]] (Val_step + Align_entail — step-level verification + adjacent-step entailment)                                                                                       | Symbolic/logical         |                                                                                |                 |
| Safety                       | [[metrics/toxicity-probability]] (TP, EMT; single-turn), [[metrics/dialogue-toxicity]] (TSG, NT2T; multi-turn)                                                                                      | Text                     |                                                                                |                 |
| Safety (agent)               | RVR (rule violation rate, aggregate + per-type), SafetySat (LTL/STL satisfaction), Rejection/Risk rates (SafeAgentBench), TTC, MDC, MSCR, Jerk_RMS, HardBrakeRate, RI, violation diagnosis accuracy | Agent                    |                                                                                |                 |
| Privacy                      | DCR, AUC_MIA, Adv_MIA, Gain_AIA                                                                                                                                                                     | Tabular                  |                                                                                |                 |
| Privacy (semi-structured)    | Graph: (ε,δ)-DP (node-level / edge-level adjacency), edge-level local DP. JSON: Recall_AON (all-or-nothing de-identification recall), Clinical Model Consistency (                                  | δ_X                      | ). Log: Sensitive Attribute Exposure, Δ_M (anonymization utility degradation). | Semi-structured |
| Robustness (semi-structured) | [[metrics/graph-atlas-distance]] (GAD, GAD^cap), Syntactic Correctness Rate (σ_SCR), Degree-distribution Deviation (D_{L2})                                                                         | Semi-structured (graph)  |                                                                                |                 |
| Robustness                   | [[metrics/reward-model-accuracy]] §Δ_OOD (in-domain vs out-of-domain reward model accuracy gap), [[metrics/strict-proof-accuracy]] §FAIRR consistency (perturbation-equivalence invariance)         | Symbolic/logical         |                                                                                |                 |
| Fairness                     | ΔSPD, ΔEO, ΔEOp                                                                                                                                                                                     | Tabular                  |                                                                                |                 |
| Provenance                   | Validation rate (traceability to original sources)                                                                                                                                                  | Vision-language          |                                                                                |                 |

### Evaluation Practice Gap Summary

| Modality | Well-covered | Under-evaluated |
|---|---|---|
| Text | (partial) Fidelity, Diversity | Validity, Faithfulness, Safety |
| Symbolic/logical | Validity | Faithfulness, Robustness |
| Tabular | Fidelity, Privacy | Fairness, Diversity |
| Semi-structured | Validity (graph/JSON) | **Privacy** (universally ×), **Utility** (mostly ×). Stem from experiment scoping, not metric-framework shortcomings. |
| Vision-language | Fidelity, Validity | Diversity, Safety, Provenance |
| Agent | Validity, Utility | Fidelity, Diversity, Diagnostic Safety |

Recurring pattern: **final-answer validity is well-covered; process-level faithfulness and distribution-level diversity are systematically neglected.**

### Open Challenges — Future Directions (Section 8)

#### §8.1 From static snapshots to dynamic feedback-loop evaluation

Most metrics in the survey (diversity via self-similarity, fidelity via distributional distance) provide a **static snapshot** of a single generation round. Practical applications are moving toward recursive training loops where synthetic data trains models that subsequently generate new data.

Static snapshots fail to capture long-term dynamics. A dataset may score high on diversity in iteration 1 yet drive **model collapse** — distributional tails disappear, modes over-amplify across train-generate cycles.

**Key direction**: move from point-wise evaluation to **longitudinal trajectory monitoring**. Dynamic metrics that track *derivatives* of quality over time: loss of support coverage, contraction of the feature space, drift of error patterns across iterations. These temporal metrics could serve as early warning signals that trigger interventions (mixing in real data, rebalancing domains, adjusting sampling strategies) before collapse becomes irreversible.

**Meta-evaluation need**: protocols to test whether existing metrics are *sufficiently sensitive* to serve as stability controllers in feedback loops. Current metrics were designed for fixed human-annotated corpora — the question is whether they carry over to the dynamic regime.

Cross-refs: [[concepts/data-repetition]] (model collapse from repetition), [[concepts/evaluation-scaling-laws]] (evaluation across scales), [[questions/evaluation]] §"Dynamic evaluation and feedback loops".

#### §8.2 Redefining fidelity: from surface mimicry to process verifiability

Fidelity has traditionally measured distributional similarity to human data. For open-ended NL tasks (dialogue), human-centric similarity remains central. For **reasoning-intensive domains**, this definition is a bottleneck — as reasoning LLMs match or surpass average human performance, enforcing adherence to human distributions penalises correct but novel solutions. In math, code, or symbolic logic, being objectively correct matters more than sounding human.

**Key direction**: fidelity metrics must evolve from measuring **mimicry** (distributional similarity to human artifacts) toward measuring **verifiability**. Scalable automated process-level rewards: execution feedback for code, formal proof checkers for mathematics, logical consistency probes for chain-of-thought traces. Assess correctness and internal coherence of the reasoning process rather than stylistic resemblance to human baselines.

Cross-refs: [[concepts/process-vs-outcome-reward]] (ORM/PRM distinction), [[concepts/verification-signals]] (execution and formal verification as V-signals), [[questions/evaluation]] §"Domain boundary for fidelity-as-verifiability".

#### §8.3 Navigating the trust–utility Pareto frontier

Quality (including downstream utility) and Trustworthiness (privacy, safety, fairness) frequently act as competing objectives during deployment. Mechanisms to increase trustworthiness impose an **alignment tax**:

- Aggressive safety filtering removes harmful content *and* disproportionately eliminates rare concepts → tail Coverage shrinks, [[concepts/diversity]] drops.
- Differential-privacy noise improves privacy guarantees but degrades predictive performance → Fidelity and Utility drop.
- Strict alignment reduces harmful outputs at the cost of elevated refusal rates on benign queries (**over-refusal**).

**Key direction**: move beyond optimising individual metrics in isolation. Characterise the **Pareto-optimal frontier** over pipeline configurations — the set of operating points where no trust metric can be improved without degrading utility or vice versa. Enable application-dependent choices: estimate how much downstream performance must be sacrificed for a target privacy guarantee ($\varepsilon, \delta$); quantify acceptable refusal-rate slack for a desired safety level. Multi-objective optimisation and selection strategies treating trust and utility as co-dependent variables, not independent checkboxes.

Cross-refs: [[concepts/alignment-tax]] (the concept note), [[concepts/trustworthiness-taxonomy]] (the second pillar), [[questions/evaluation]] §"Trust-utility Pareto frontier".

## Core Concepts

- Extracted from this paper (text modality, Phase 1 — evaluation side):
  - [[concepts/intrinsic-vs-extrinsic-evaluation]] — the paper's central meta-framing: measure the data directly, before training, as a complement to downstream extrinsic verdicts.
  - [[concepts/trustworthiness-taxonomy]] — the second pillar alongside quality. Full text-modality content (Faithfulness, Safety); other five dimensions stubbed.
  - [[concepts/alignment-tax]] — the trust-utility Pareto trade-off from §8.3: safety filtering shrinks diversity, DP noise degrades fidelity, alignment raises refusal rates.
- Extracted from this paper (Phase 3 — generation side, §2.1):
  - [[concepts/generation-intervention-loci]] — the §2.1 locus-of-intervention taxonomy (Source-Corpus Control / Prompt-Driven / Alignment-Based / Inference-Time), mapped onto the $(S, M, G, f, V)$ tuple from [[concepts/synthetic-data-formalism]]. Sits orthogonal to the grounding-axis landscape.
- Extracted from this paper (Phase 4 — symbolic / logical data, §3):
  - [[concepts/reasoning-data-generation]] — §3.1's four generation archetypes (heuristic evolution, tool-verified synthesis, rewarded-rollout harvesting, preference curation) as reasoning-domain specialisations of the four loci.
  - [[concepts/process-vs-outcome-reward]] — the ORM / PRM distinction from §3.2 and §3.3, placed in the verification-signals spine.
  - [[concepts/reasoning-chain-judges]] — reasoning-specific LLM-judge variant; anchors the judge–human correlation metric.
- Existing concepts extended or cross-linked:
  - [[concepts/evaluation-targets]] — amended to split Quality into Validity + Fidelity + Quality (composite), with a mapping table back to the Auditor taxonomy.
  - [[concepts/evaluation-lifecycle]] — amended with an "Intrinsic vs extrinsic across the lifecycle" section.
  - [[concepts/verification-signals]] — Val_step, PassRate, Acc_verify are verification signals; this paper provides a taxonomy across modalities.
  - [[concepts/llm-as-judge-methodology]] — ρ_LLM-human, Acc_RM, and the fidelity-of-proxy discussion directly address judge reliability.
  - [[concepts/critic-validation]] — faithfulness metrics (Attr_AIS, Align_entail) formalise what critic validation should measure.
  - [[concepts/multi-property-data-curation]] — the quality/trustworthiness two-pillar taxonomy is a framework for multi-property curation.
  - [[concepts/data-repetition]] — model collapse discussion (§8.1) connects to data repetition concerns.
  - [[concepts/evaluation-scaling-laws]] — the static-vs-dynamic evaluation gap is about evaluation under scaling.
- Existing concepts extended (Phase 6 — agent data, §7):
  - [[concepts/synthetic-data-formalism]] — added §Grounding extensions: interactive environments (environment as seed, three sub-shapes).
  - [[concepts/trajectory-synthesis]] — added §Agent-data product taxonomy (three data-product categories mapped to the trajectory formalism).
  - [[concepts/evaluation-targets]] — added §Agent-data instantiations (per-target mapping for agent data).
  - [[concepts/trustworthiness-taxonomy]] — filled §Agent-modality: Safety (RVR, SafetySat, Rejection/Risk, TTC, MDC, comfort metrics).
  - [[concepts/verification-signals]] — added simulator execution and LTL/STL model checker to domain-checker spectrum.
  - [[maps/evaluation/landscape]] — added agent-data note and expanded trustworthiness Safety row.
  - [[maps/grounding/landscape]] — added environment-grounding cross-reference.
- Existing concepts extended (Phase 7 — future directions, §8):
  - [[concepts/alignment-tax]] — expanded with §8.3's multi-objective optimisation framing and application-dependent Pareto selectors.
  - [[concepts/data-repetition]] — cross-referenced to §8.1's recursive-loop collapse concern.
  - [[concepts/evaluation-scaling-laws]] — cross-referenced to §8.1's dynamic evaluation challenge.
  - [[concepts/process-vs-outcome-reward]] — cross-referenced to §8.2's fidelity-as-verifiability argument.
  - [[questions/evaluation]] — three new question clusters from §8 directions.

## Relevance To Poolside

*Our interpretation:*

1. **Synthetic data quality audit framework.** The quality/trustworthiness taxonomy gives a structured checklist for evaluating Poolside's synthetic data pipelines. The gap tables show what most teams skip — faithfulness and diversity metrics in particular. Poolside's data pipelines could be audited against this taxonomy.

2. **Code/reasoning data metrics.** Section 3 (symbolic/logical) formalises exactly the metrics relevant to Poolside's code and math synthetic data: PassRate, Acc_verify, Val_step (step-level faithfulness), Align_entail. These map directly to verification signals we already think about.

3. **Process-level fidelity for reasoning.** The Section 8.2 argument — that fidelity should shift from "sounds human" to "is verifiably correct" — aligns with Poolside's investment in execution-verified synthetic data. This validates the approach of treating test execution, proof checking, and logical consistency as primary fidelity signals.

4. **Diversity metrics for mode collapse.** Self-CosSim, Distinct-N, and entropy metrics provide concrete operationalisations for measuring whether synthetic data generators are producing diverse outputs or collapsing into repetitive patterns. Directly applicable to monitoring rephrasing and instruction-generation pipelines.

5. **Dynamic evaluation for recursive loops.** The Section 8.1 argument for longitudinal trajectory monitoring is relevant if Poolside uses model-generated data to train subsequent models — need metrics that detect distributional drift across generations.

6. **Faithfulness gap is the biggest gap.** Across all modalities, intermediate-step faithfulness is the most neglected dimension. For code reasoning data, this means: a model can produce correct final answers via unfaithful reasoning chains, and most pipelines wouldn't catch it.

## Blaz Notes
- Especially the section on reasoning data is a very good reference, overall the paper manages to cover many ideas and wrap them up into a nice point of view which will be useful for our follow-up work.

## Key Follow-Ups / Jumping-Off Points

1. **Audit Poolside synth-data pipelines against Table 3** (symbolic/logical gap analysis). Which of {validity, fidelity, faithfulness, robustness} do our pipelines currently measure? Where are we ×?
2. **Operationalise Val_step for code reasoning.** Step-level verification of chain-of-thought in code problems — can we use execution traces or intermediate assertions as step verifiers?
3. **Self-CosSim as a diversity monitor.** Implement embedding-based diversity tracking for synthetic data batches to detect mode collapse early.
4. **Longitudinal metrics for recursive pipelines.** If we train on model-generated data, track quality/diversity metrics across generations, not just within a single generation round.
5. **Process-level fidelity vs answer-level validity.** The paper's central argument: validating final answers is necessary but not sufficient. Faithfulness of intermediate reasoning is where current approaches fail. Implications for how we filter/curate reasoning traces.
6. **Trust-utility trade-off measurement.** When applying safety filtering or decontamination to synthetic data, measure what we lose in diversity/coverage, not just what we gain in safety.
7. **Read referenced papers.** Key references to follow up: RARR (Gao et al., 2023a) for attribution metrics, ReCEval (Prasad et al., 2023) for step-level reasoning evaluation, FRODO (Paul et al., 2024) for causal faithfulness of chain-of-thought.

## Related Notes

- Concepts (new, from this paper): [[concepts/intrinsic-vs-extrinsic-evaluation]], [[concepts/trustworthiness-taxonomy]], [[concepts/alignment-tax]], [[concepts/generation-intervention-loci]], [[concepts/reasoning-data-generation]], [[concepts/process-vs-outcome-reward]], [[concepts/reasoning-chain-judges]]
- Concepts (existing, cross-linked): [[concepts/evaluation-targets]], [[concepts/evaluation-lifecycle]], [[concepts/verification-signals]], [[concepts/synthetic-data-formalism]], [[concepts/llm-as-judge-methodology]], [[concepts/critic-validation]], [[concepts/multi-property-data-curation]], [[concepts/data-filtering-paradigms]], [[concepts/data-repetition]], [[concepts/evaluation-scaling-laws]], [[concepts/downstream-utility-as-ground-truth]], [[concepts/trajectory-synthesis]], [[concepts/pass-at-k-methodology]]
- Maps: [[maps/evaluation/landscape]], [[maps/grounding/landscape]]
- Papers (Phase 4 stubs created): [[notes/papers/2024-metamath]], [[notes/papers/2024-openmathinstruct-1]], [[notes/papers/2021-proofwriter]], [[notes/papers/2025-synlogic]], [[notes/papers/2022-fairr]], [[notes/papers/2023-receval]]
- Papers (Phase 5 — semi-structured stubs created): [[notes/papers/2025-schemabench]], [[notes/papers/2025-think-inside-the-json]], [[notes/papers/2025-jsonschemabench]], [[notes/papers/2024-llm4graphgen]], [[notes/papers/2025-llms-prompted-for-graphs]], [[notes/papers/2024-logbench]]
- Papers (Phase 6 — agent data stubs created): [[notes/papers/2024-partnr]], [[notes/papers/2024-selp]], [[notes/papers/2025-t3-planner]], [[notes/papers/2024-safeagentbench]], [[notes/papers/2025-world-models-survey]], [[notes/papers/2025-ttsg]], [[notes/papers/2022-saycan]], [[notes/papers/2023-code-as-policies]], [[notes/papers/2024-eureka]], [[notes/papers/2024-vbench]]
- Papers (referenced by this survey, not yet enriched — arXiv-inline only): RARR (Gao et al., 2023a, arXiv [2210.08726](https://arxiv.org/abs/2210.08726)) — attribution and preservation metrics; FRODO (Paul et al., 2024, arXiv [2402.14317](https://arxiv.org/abs/2402.14317)) — causal faithfulness of chain-of-thought; OpenCodeInstruct (Ahmad et al., 2025, arXiv [2504.04030](https://arxiv.org/abs/2504.04030)) — large-scale code instruction tuning; WizardMath (arXiv [2308.09583](https://arxiv.org/abs/2308.09583)); WizardCoder (arXiv [2306.08568](https://arxiv.org/abs/2306.08568)); DeepSeek-R1 (arXiv [2501.12948](https://arxiv.org/abs/2501.12948)); RewardBench (arXiv [2403.13787](https://arxiv.org/abs/2403.13787)); JudgeLM (arXiv [2310.17631](https://arxiv.org/abs/2310.17631)); RLAIF (arXiv [2309.00267](https://arxiv.org/abs/2309.00267)); STaR (arXiv [2203.14465](https://arxiv.org/abs/2203.14465)); Let's Verify Step-by-Step (arXiv [2305.20050](https://arxiv.org/abs/2305.20050)); Dataflow (Liang et al., 2025) — generate-evaluate-filter-refine paradigm

## Caveats

- **Survey, not experimental paper.** No novel experiments; the contribution is taxonomic. Metric definitions are assembled from cited works, not validated in new settings.
- **Metric selection is non-exhaustive.** The authors acknowledge their metric collection is not complete. Some modalities (agent, vision-language) have sparser metric coverage than text and tabular.
- **Gap analysis is based on representative methods, not a systematic review.** Tables 2–7 audit a handful of methods per modality, not the full literature. A method not included might cover dimensions the table marks as missing.
- **No empirical validation of the taxonomy itself.** The paper doesn't test whether auditing against this framework actually predicts downstream training outcomes.
- **Trustworthiness dimensions are unevenly developed.** Privacy and fairness are well-formalised for tabular data but barely exist for text and reasoning modalities. Provenance is only covered for vision-language.
