---
arxiv: '2604.06829'
authors:
- Jiang Zhou
- Yunhao Wang
- Xing Wu
- Tinghao Yu
- Feng Zhang
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'WRAP++: Web discoveRy Amplified Pretraining'
url: https://arxiv.org/abs/2604.06829
year: 2026
---

[2604.06829] WRAP++: Web Discovery Amplified Pretraining














function detectColorScheme(){
var theme="light";
var current\_theme = localStorage.getItem("ar5iv\_theme");
if(current\_theme){
if(current\_theme == "dark"){
theme = "dark";
} }
else if(!window.matchMedia) { return false; }
else if(window.matchMedia("(prefers-color-scheme: dark)").matches) {
theme = "dark"; }
if (theme=="dark") {
document.documentElement.setAttribute("data-theme", "dark");
} else {
document.documentElement.setAttribute("data-theme", "light"); } }
detectColorScheme();
function toggleColorScheme(){
var current\_theme = localStorage.getItem("ar5iv\_theme");
if (current\_theme) {
if (current\_theme == "light") {
localStorage.setItem("ar5iv\_theme", "dark"); }
else {
localStorage.setItem("ar5iv\_theme", "light"); } }
else {
localStorage.setItem("ar5iv\_theme", "dark"); }
detectColorScheme(); }



# WRAP++: Web Discovery Amplified Pretraining

Jiang Zhou
  
Yunhao Wang
  
Xing Wu
  
Tinghao Yu
  
Feng Zhang

###### Abstract

Synthetic data rephrasing has emerged as a powerful technique for enhancing knowledge acquisition during large language model (LLM) pretraining.
However, existing approaches operate at the *single-document* level, rewriting individual web pages in isolation.
This confines synthesized examples to intra-document knowledge, missing cross-document relationships and leaving facts with limited associative context.
We propose WRAP++ (Web discoveRy Amplified Pretraining), which *amplifies* the associative context of factual knowledge by *discovering* cross-document relationships from web hyperlinks and synthesizing joint QA over each discovered document pair.
Concretely, WRAP++ discovers high-confidence relational motifs including dual-links (A↔BA\leftrightarrow B) and co-mentions (A→E←BA\rightarrow E\leftarrow B with A→BA\rightarrow B), and synthesizes QA that requires reasoning across both documents. This produces relational knowledge absent from either source document alone, creating diverse entry points to the same facts.
Because the number of valid entity pairs grows combinatorially, this discovery-driven synthesis also amplifies data scale far beyond single-document rewriting.
Instantiating WRAP++ on Wikipedia, we amplify ∼\sim8.4B tokens of raw text into 80B tokens of cross-document QA data. On SimpleQA, OLMo-based models at both 7B and 32B scales trained with WRAP++ substantially outperform single-document approaches and exhibit sustained scaling gains, underscoring the advantage of cross-document knowledge discovery and amplification.

††footnotetext: † Corresponding Author. Correspondence to ucaswu@tencent.com.††footnotetext: 

## 1 Introduction

Synthetic data has become an increasingly important component of large language model (LLM) pretraining. WRAP (maini2024wrap) showed that rephrasing noisy web text into QA format can improve pretraining, and later systems scaled this recipe substantially: Nemotron-CC (su-etal-2025-nemotron) produced ∼\sim2 trillion synthetic tokens from Common Crawl, Phi-4 (abdin2024phi4technicalreport) used 40% synthetic data in pretraining, and Qwen3 (yang2025qwen3) incorporated synthetic data into its training pipeline.

However, this progress has been explored mainly along intra-document axes—rephrasing strategy, generator model, and source quality (nguyen2025recycling; niklaus2026finephrase)—varying *how* a single document is rewritten without changing *what* is synthesized.
Because many facts are distributed across multiple documents, this single-document paradigm confines the model to limited associative context for each fact, which ultimately hinders knowledge recoverability.

This limitation motivates a shift toward *cross-document synthesis*: bringing multiple facts into a shared context to learn relational knowledge jointly.
However, moving to cross-document synthesis is non-trivial.
A naïve approach of randomly pairing documents yields little improvement over single-document baselines (see §[4.1](#S4.SS1 "4.1 Necessity of Topological Relation Discovery ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")), as forcing an LLM to synthesize joint QA from unrelated texts produces fabricated connections and low-quality data.
Thus, the *document selection* mechanism is critical: cross-document synthesis only succeeds when the paired documents contain genuinely related facts.

Web hyperlinks provide a broad relevance signal by encoding human-curated judgments of importance (zhou2022hyperlink).
For example, the Wikipedia pages of composers Hans Zimmer and Ludwig Göransson are topologically linked through shared collaborations with director Christopher Nolan. While single-document synthesis might only extract isolated facts (e.g., ”Göransson won an Oscar for Oppenheimer”), WRAP++ pairs these connected documents to synthesize multi-hop relational QA. As illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ WRAP++: Web Discovery Amplified Pretraining"), the model is forced to explicitly reason across both texts—deducing that Zimmer left Tenet to score Dune (his second Oscar), leading to Göransson’s hiring and subsequent second Oscar for Oppenheimer. This explicit relational supervision provides vital disambiguation cues and diverse retrieval paths absent from single-document synthesis, saving the LLM from having to implicitly infer these complex connections from massive unstructured text.

![Refer to caption](/html/2604.06829/assets/x1.png)


Figure 1: Overview of the WRAP++ pipeline. Unlike single-document WRAP, which rewrites individual documents, WRAP++ *discovers* cross-document entity relationships from web topology and *amplifies* them into pretraining data through joint QA synthesis.

Based on this insight, we propose WRAP++ (Web discovRy Amplified Pretraining), which effectively extends the synthetic data paradigm from single-document *rewriting* to cross-document *discovery and amplification*.

Amplifying Associative Context via Relation Discovery. We discover high-confidence relational motifs from web hyperlinks (zhou2022hyperlink)—specifically *dual-links* (A↔BA\leftrightarrow B) and *co-mentions* (A→E←BA\rightarrow E\leftarrow B with A→BA\rightarrow B). Rather than simply concatenating these documents, we feed the discovered pairs to an instruction-tuned LLM generator subjected to three strict synthesis constraints: enforcing Cross-Document Dependency to mandate joint reasoning, requiring Explicit Factual Chaining to decode multi-hop logical paths, and ensuring Omniscient Internalization by forbidding local document attribution. This process produces genuinely new relational knowledge (comparisons, contrasts, bridging facts) that creates diverse retrieval paths to the same facts. Furthermore, because the number of valid entity pairs grows combinatorially, this discovery-driven synthesis achieves a ∼\sim10×\times data amplification—scaling a fixed 8.4B-token source corpus into 80B tokens of cross-document QA data—consistently improving the knowledge recoverability of the model.

We instantiate WRAP++ on Wikipedia and amplify ∼\sim8.4B tokens of raw text into 80B tokens of cross-document QA data—compared to only ∼\sim5.4B tokens from single-document WRAP.
On the SimpleQA benchmark (wei2024simpleqa), OLMo-based models at both 7B and 32B scales trained with WRAP++ data substantially outperform all single-document baselines, and WRAP++ demonstrates a more favorable scaling trajectory than single-document approaches.

Our contributions are three-fold:

1. 1.

   We propose WRAP++, a framework that extends single-document rewriting into topology-guided relation discovery and joint QA synthesis, amplifying the associative context of factual knowledge.
2. 2.

   We instantiate WRAP++ on Wikipedia to synthesize 80B tokens of cross-document QA data, demonstrating that combinatorial relation discovery enables data amplification far beyond single-document synthesis.
3. 3.

   We show on SimpleQA with OLMo-based 7B and 32B models that WRAP++ substantially outperforms single-document baselines, exhibiting a favorable scaling trajectory.

## 2 Method: WRAP++

WRAP++ is a framework that transitions synthetic pretraining data from single-document rewriting to topology-guided cross-document discovery and amplification. The framework consists of two core stages: Topological Relation Discovery (§[2.2](#S2.SS2 "2.2 Topological Relation Discovery ‣ 2 Method: WRAP++ ‣ WRAP++: Web Discovery Amplified Pretraining")) and Joint QA Synthesis (§[2.3](#S2.SS3 "2.3 Cross-Document Joint QA Synthesis ‣ 2 Method: WRAP++ ‣ WRAP++: Web Discovery Amplified Pretraining")). Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ WRAP++: Web Discovery Amplified Pretraining") illustrates the overall pipeline.

### 2.1 Problem Formulation and Graph Abstraction

Let 𝒟={d1,d2,…,dN}\mathcal{D}=\{d\_{1},d\_{2},\dots,d\_{N}\} denote a large-scale web corpus consisting of NN documents. The inter-document references (e.g., hyperlinks) naturally induce a directed knowledge graph 𝒢=(𝒱,ℰ)\mathcal{G}=(\mathcal{V},\mathcal{E}), where each vertex vi∈𝒱v\_{i}\in\mathcal{V} corresponds to a document did\_{i}, and a directed edge ei,j∈ℰe\_{i,j}\in\mathcal{E} exists if did\_{i} explicitly references djd\_{j}. Since each document in our corpus describes a single entity, we use “entity pair” and “document pair” interchangeably throughout this paper.

Conventional single-document synthesis operates solely on the local context of viv\_{i}, limiting the model’s exposure to isolated facts. In contrast, WRAP++ leverages the topological structure of 𝒢\mathcal{G} to discover genuine semantic dependencies across documents, bringing related knowledge into a shared synthesis context to amplify the associative context of factual knowledge.

### 2.2 Topological Relation Discovery

A naïve approach of pairing random documents from 𝒟\mathcal{D} forces the synthesis model to hallucinate spurious connections. To ensure the semantic validity of cross-document synthesis, we discover high-confidence relational motifs directly from 𝒢\mathcal{G}. We focus on two topological structures that provide strong inductive biases for relational reasoning:

#### Dual-link Motif.

Two documents uu and vv form a dual-link relationship if they mutually reference each other. Formally, a dual-link pair (u,v)(u,v) satisfies the bidirectional constraint:

|  |  |  |  |
| --- | --- | --- | --- |
|  | eu,v∈ℰ∧ev,u∈ℰe\_{u,v}\in\mathcal{E}\land e\_{v,u}\in\mathcal{E} |  | (1) |

This mutual dependency typically indicates a strong, foundational semantic correlation (e.g., a notable director and their magnum opus, or a scientist and their core discovery). Discovering this motif ensures the underlying entity pair is highly coupled.

#### Co-mention Motif.

Documents uu and vv share a co-mention relationship if they both reference a common structural hub EE, while maintaining a direct link between themselves. Formally, the triplet (u,v,E)(u,v,E) satisfies:

|  |  |  |  |
| --- | --- | --- | --- |
|  | eu,E∈ℰ∧ev,E∈ℰ∧eu,v∈ℰe\_{u,E}\in\mathcal{E}\land e\_{v,E}\in\mathcal{E}\land e\_{u,v}\in\mathcal{E} |  | (2) |

The shared structural context EE imposes implicit analogical, hierarchical, or comparative relationships (e.g., two competing theories cited in the same survey article). This motif explicitly encourages the subsequent synthesis model to generate relational knowledge that contrasts and compares the related entities, thereby amplifying their associative context.

### 2.3 Cross-Document Joint QA Synthesis

Given a discovered document pair (du,dv)(d\_{u},d\_{v}) connected by a valid topological motif, we employ an instruction-tuned LLM generator ℳθ\mathcal{M}\_{\theta} to synthesize a set of composite QA instances 𝒮u,v={(qi,ci,ai)}i=1K\mathcal{S}\_{u,v}=\{(q\_{i},c\_{i},a\_{i})\}\_{i=1}^{K}, where qiq\_{i} is the question, cic\_{i} is the intermediate factual chain, and aia\_{i} is the final answer.

The generation process is conditioned on a structured prompt 𝒫\mathcal{P} and the concatenated document context:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮u,v∼ℳθ​(du⊕dv,𝒫)\mathcal{S}\_{u,v}\sim\mathcal{M}\_{\theta}(d\_{u}\oplus d\_{v},\mathcal{P}) |  | (3) |

To amplify the associative context of the synthesized data and prevent the generator from degrading to shallow single-document summarization, 𝒫\mathcal{P} enforces three functional constraints on the output space:

* •

  Strict Cross-Document Dependency: The generated question qiq\_{i} must have high entropy given only one document. Deriving the correct answer aia\_{i} must strictly require logical premises from both dud\_{u} and dvd\_{v}, ensuring the synthesis produces genuinely new relational knowledge rather than merely rephrasing isolated facts.
* •

  Explicit Factual Chaining: Before outputting aia\_{i}, the generator must explicitly decode the traversal path cic\_{i}. By articulating the necessary facts extracted from both documents and linking them step-by-step, the pretraining model internalizes multi-hop knowledge structures, effectively creating diverse associative entry points to the underlying facts.
* •

  Omniscient Internalization: The generator is strictly prohibited from attributing facts to the local context (e.g., avoiding “According to Passage A”). It must output universally valid statements. This ensures the synthesized data serves as parametric world knowledge rather than context-dependent reading comprehension exercises.

## 3 Experiments

### 3.1 Experimental Setup

#### Synthesis Data.

In principle, WRAP++ is a general framework applicable to any text corpus containing hyperlinks. In this work, we instantiate it on Wikipedia because of its rich link structure and its widespread use in prior work on synthetic rewriting methods (maini2024wrap; su-etal-2025-nemotron). Specifically, we use the English subset of FineWiki (penedo2025finewiki) as our base corpus (𝒟\mathcal{D}), which contains approximately 8.4B tokens. We parse the hyperlinks in FineWiki to construct the directed inter-document graph 𝒢\mathcal{G} used for topological relation discovery.

#### Synthesis Model.

We use Qwen3-30B-A3B-Instruct-FP8 as our instruction-tuned generator ℳθ\mathcal{M}\_{\theta}. The prompt is designed to enforce the strict cross-document dependency and explicit factual chaining constraints described in §[1](#S1 "1 Introduction ‣ WRAP++: Web Discovery Amplified Pretraining"), thereby encouraging high-quality relational QA generation. The full prompt template is provided in Appendix [D](#A4 "Appendix D Synthesis Prompt Templates ‣ WRAP++: Web Discovery Amplified Pretraining").

#### Data Scale.

Topological relation discovery substantially amplifies the data scale beyond individual documents. The dual-link motif yields highly coupled entity pairs that produce ∼\sim3B tokens of cross-document QA data. Incorporating the co-mention motif broadens coverage, bringing the combined WRAP++ dataset to ∼\sim82.7B tokens.

#### Training Models.

To assess the effect of cross-document synthetic data on parametric knowledge, we continue pretraining from the OLMo-3 stage-1 last checkpoint at both the 7B and 32B scales for one epoch. We choose OLMo because it is fully open-source and provides publicly released checkpoints throughout training, making it a suitable platform for controlled continued-pretraining experiments.

### 3.2 Evaluation Setup

#### Benchmark.

We evaluate on SimpleQA (wei2024simpleqa), a knowledge-intensive benchmark designed to measure short-form factual accuracy while minimizing sensitivity to formatting heuristics. Most SimpleQA questions can be answered directly from Wikipedia-derived knowledge, making it a natural testbed for studying factual knowledge acquisition under our setup.

#### Metric.

We use pass@128 as our primary metric, defined as the empirical probability that at least one of 128 sampled responses contains the correct fact. Our goal is to measure *knowledge recoverability* rather than only top-1 answer accuracy. In this setting, pass@128 is useful because it probes whether the correct factual association can be elicited from the model under sampling, even when it is not the single most likely surface form. We therefore treat it as a more sensitive indicator of parametric knowledge recoverability during continued pretraining than pass@1.

#### Baselines.

To isolate the benefit of cross-document synthesis, we compare WRAP++ against two closely related single-document baselines derived from the same FineWiki corpus:

1. 1.

   WRAP (∼\sim5.4B tokens): Standard single-document QA synthesis following the original WRAP recipe (maini2024wrap), representing the typical yield of intra-document fact extraction.
2. 2.

   Extended WRAP (∼\sim17.4B tokens): An expanded single-document synthesis utilizing additional prompting strategies (e.g., exhaustive extraction) to push the limits of single-document scaling.

These baselines also illustrate the data-scaling constraint of single-document synthesis: on the same FineWiki corpus, standard WRAP yields only ∼\sim5.4B tokens and Extended WRAP reaches ∼\sim17.4B tokens, both well below the ∼\sim82.7B-token scale of WRAP++.

### 3.3 Main Results

Table 1: SimpleQA results after 1-epoch continued training on OLMo-3-7B and OLMo-3-32B using different data recipes. The metric reported is the empirical pass@128 rate (%).

| Data Recipe | OLMo-3-7B | OLMo-3-32B |
| --- | --- | --- |
| Pretrained Base | 34.76 | 42.35 |
| + WRAP | 39.55 | 44.43 |
| + Extended WRAP | 43.69 | 47.91 |
| + WRAP++ | 49.13 | 53.97 |

Table [1](#S3.T1 "Table 1 ‣ 3.3 Main Results ‣ 3 Experiments ‣ WRAP++: Web Discovery Amplified Pretraining") presents the main results of 1-epoch continued training across different synthesis recipes. We highlight two principal findings.

#### WRAP++ substantially outperforms single-document baselines.

Across both model scales, continued pretraining with WRAP++ yields substantially higher pass@128 on SimpleQA compared to all single-document approaches (+9.5 pp on 7B, +9.8 pp on 32B over WRAP; +5.4 pp on 7B, +6.1 pp on 32B over Extended WRAP).
This advantage reflects two complementary factors.
First, cross-document synthesis produces higher-quality relational knowledge *per token*: at a matched budget of ∼\sim8B tokens, WRAP++ already outperforms Extended WRAP by +2.48 pp (detailed in §[4.5](#S4.SS5 "4.5 Comparison with Other Single-Document Strategies ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")), confirming a genuine quality advantage independent of data scale.
Second, the combinatorial nature of relation discovery amplifies this quality advantage to a far larger data space (∼\sim80B tokens) that single-document methods cannot access, yielding further gains as training progresses (Figure [2](#S3.F2 "Figure 2 ‣ 3.4 Scaling and Training Dynamics ‣ 3 Experiments ‣ WRAP++: Web Discovery Amplified Pretraining")).

#### Surpassing the single-document scaling bottleneck.

Single-document methods face an inherent data bottleneck: the finite number of extractable facts within an individual page. While Extended WRAP attempts to push this limit through exhaustive extraction (reaching ∼\sim17.4B tokens), it ultimately depletes the source material. The resulting diminishing returns (+4.1 pp on 7B, +3.7 pp on 32B over standard WRAP) suggest information saturation under the single-document paradigm.
In contrast, because the number of valid cross-document entity pairs grows combinatorially, WRAP++ amplifies the same FineWiki source corpus into ∼\sim80B tokens of relational knowledge—a data space fundamentally inaccessible to single-document methods. Single-document methods *cannot* close this gap simply by training longer, since their source material is already exhausted. We analyze the resulting scaling dynamics in detail next.

### 3.4 Scaling and Training Dynamics

![Refer to caption](/html/2604.06829/assets/x2.png)


Figure 2: SimpleQA pass@128 vs. training tokens. Single-document recipes (WRAP and Extended WRAP) reach a data bottleneck early, limiting further knowledge acquisition. In contrast, the combinatorial nature of WRAP++ allows it to scale effectively up to 80B tokens, improving performance without obviously plateauing.

![Refer to caption](/html/2604.06829/assets/x3.png)


Figure 3: Evolution of pass@kk performance during training. The curves illustrate the unbiased pass@kk of OLMo-3-7B (a) and OLMo-3-32B (b) on SimpleQA (k∈[1,128]k\in[1,128] in log scale). The color gradient (from light to dark blue) tracks the accumulation of consumed WRAP++ tokens (from 10B to 80B). The strictly monotonic upward shift across all values of kk indicates robust, unsaturated knowledge internalization.

Figure [2](#S3.F2 "Figure 2 ‣ 3.4 Scaling and Training Dynamics ‣ 3 Experiments ‣ WRAP++: Web Discovery Amplified Pretraining") plots pass@128 as a function of training tokens consumed. The trajectories confirm the scaling bottleneck discussed above: single-document recipes plateau early, whereas WRAP++ maintains a steady upward trend all the way to 80B tokens without obvious saturation, demonstrating that the combinatorial data space opened by relation discovery translates into sustained knowledge gains.

To further dissect how this scaling translates into knowledge recoverability, we track the evolution of the pass@kk curves throughout training. Figure [3](#S3.F3 "Figure 3 ‣ 3.4 Scaling and Training Dynamics ‣ 3 Experiments ‣ WRAP++: Web Discovery Amplified Pretraining") visualizes the unbiased SimpleQA pass@kk for both models. As training progresses (indicated by the light-to-dark blue gradient representing the 80B token influx), the curves exhibit a monotonic upward shift across all values of kk.

Crucially, this improvement spans the entire logarithmic kk-spectrum. The persistent lift at small kk (the leftmost regions of the curves) shows that the model’s top-ranked answers increasingly contain the correct fact, reflecting higher precision. Simultaneously, the parallel gains at larger kk indicate a broader and more robust set of associative retrieval paths to the same knowledge.

## 4 Ablations and Analysis

We conduct extensive ablations to validate each component of WRAP++.
Due to experimental costs, unless otherwise noted, all ablation experiments use OLMo-3-7B continued pretraining with ∼\sim8B tokens and report the results in terms of SimpleQA pass@128.

Table 2: Ablation results across different design choices (OLMo-3-7B, ∼\sim8B tokens). We report SimpleQA pass@128. WRAP++ (default) uses topological relation discovery, combined topologies, joint QA synthesis, and the Qwen3-30B-A3B synthesis model.

|  |  |  |
| --- | --- | --- |
| Ablation Axis | Variant | pass@128 |
| Pairing Strategy (§[4.1](#S4.SS1 "4.1 Necessity of Topological Relation Discovery ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")) | Random entity pairing | 43.46 |
| Topological relation discovery | 45.11 |
| Relation Topology (§[4.2](#S4.SS2 "4.2 Topology Comparison: Dual-Link vs. Co-Mention ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")) | Dual-link only | 44.24 |
| Co-mention only | 44.36 |
| Combined | 45.11 |
| Synthesis Method (§[4.3](#S4.SS3 "4.3 Necessity and Format of QA Synthesis ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")) | Raw concatenation (no QA) | 35.43 |
| QA with source documents prepended | 38.93 |
| Joint QA synthesis | 45.11 |
| Synthesis Model Scale (§[4.4](#S4.SS4 "4.4 Effect of Synthesis Model Scale ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")) | Qwen3-30B-A3B | 45.11 |
| Qwen3-235B-A22B | 47.70 |

### 4.1 Necessity of Topological Relation Discovery

We explore whether the topological relation discovery is essential, or whether randomly pairing Wikipedia pages would suffice.
Table [2](#S4.T2 "Table 2 ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") (Pairing Strategy rows) shows a clear performance drop (from 45.11 to 43.46) when entities are paired randomly rather than via dual-link or co-mention relations.
Qualitatively, random pairing forces the synthesis model to fabricate relationships between unrelated entities, producing factually incorrect comparisons and superficial connections.
This confirms that principled relation discovery—specifically, dual-link and co-mention motif discovery—is important for high-quality cross-document synthesis.

### 4.2 Topology Comparison: Dual-Link vs. Co-Mention

We explore the contribution of each relation type at a matched token budget of ∼\sim8B.
As shown in Table [2](#S4.T2 "Table 2 ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") (Relation Topology rows), both topologies provide strong relational signal at this budget. Co-mention retains a slight edge over dual-link (44.36 vs. 44.24), while their combination yields the best overall performance (45.11). This suggests that bidirectional links and shared structural context capture yet complementary aspects of cross-document knowledge.

### 4.3 Necessity and Format of QA Synthesis

We explore the optimal data format for learning cross-document relationships by comparing our joint QA synthesis against two alternatives: (1) raw concatenation of related documents (no QA), and (2) prepending source documents to the synthesized QA pairs. Table [2](#S4.T2 "Table 2 ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") shows raw concatenation performs only marginally above the pretrained base (35.43 vs. 34.8), indicating that explicit *synthesis* is essential to convert document proximity into learnable relational knowledge. Moreover, prepending source documents to QA pairs causes a notable performance drop (38.93), likely by allowing the model to superficially copy answers rather than parametrically internalizing them. Thus, joint QA synthesis provides the most effective format for amplifying associative context.

### 4.4 Effect of Synthesis Model Scale

We explore the effect of synthesis model scale on WRAP++ quality by comparing two synthesis models of different scales: Qwen3-30B-A3B (3B active parameters) and Qwen3-235B-A22B (22B active parameters).
Table [2](#S4.T2 "Table 2 ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") (Synthesis Model Scale rows) shows that the larger model produces higher-quality cross-document QA, leading to better downstream pass@128.
In practice, the choice between synthesis models involves a cost–quality tradeoff: the larger model is preferable when generation budget is not the bottleneck, while the smaller model enables broader coverage at lower compute cost.

Table 3: Performance comparison of WRAP++ mixed with other single-document strategies. Each mixture contains ∼\sim8B tokens total.

|  |  |  |
| --- | --- | --- |
| Single-Document Component | Proportion of WRAP++ in Mixture | |
| 0% (Baseline Only) | 50% (1:1 Mix) |
| Pretrained Model (No further training) | 34.76 | |
| Raw FineWiki | 39.23 | 41.80 |
| Distill | 38.12 | 41.59 |
| Extract Knowledge | 38.69 | 43.35 |
| Knowledge List | 38.60 | 42.03 |
| WRAP++ (100%, Ours) | 45.11 | |

### 4.5 Comparison with Other Single-Document Strategies

We contextualize the performance of WRAP++ against other representative single-document rephrasing strategies su-etal-2025-nemotron applied to the identical FineWiki source corpus, including:
(a) Distill—rewriting into cleaner, more concise prose while preserving information;
(b) Extract Knowledge—extracting key factual statements and discarding redundancy;
(c) Knowledge List—outputting structured knowledge in list format.

Table [3](#S4.T3 "Table 3 ‣ 4.4 Effect of Synthesis Model Scale ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") presents both the isolated performance (0% and 100%) and the mixing dynamics (50% blending) at a strictly restricted budget of ∼\sim8B tokens.
When evaluated in isolation at the 8B-token budget, pure WRAP++ (45.11) substantially outperforms all listed single-document baselines. This margin at a restricted data scale reveals an important dynamic in pretraining efficiency: discovery-driven synthesis already yields stronger knowledge recoverability, even before exploiting its data amplification headroom.
When blending WRAP++ with other strategies in a 1:1 ratio, we observe a clear *uplift effect*: injecting WRAP++ into any weaker baseline consistently improves upon its 0% counterpart (e.g., Knowledge List rises from 38.60 to 42.03), further highlighting the advantage of discovery-driven synthesis.

### 4.6 Integration with OLMo-3 Mid-Training Data

Table 4: Integrating WRAP++ into OLMo-3-7B 100B-token mid-training. We report SimpleQA pass@128 and the average over 12 general benchmarks (including MMLU Redux, HellaSwag, etc. See Appendix [F](#A6 "Appendix F Mid-Training Evaluation Benchmarks ‣ WRAP++: Web Discovery Amplified Pretraining") for the full list of tasks).

|  |  |  |
| --- | --- | --- |
| Setting | SimpleQA | Gen. Avg |
|  | pass@128 | (12 tasks) |
| Pretrained Base | 34.76 | 57.79 |
| + Midtrain (100B) | 34.74 | 68.24 |
| + WRAP++ Mix (100B) | 37.58 | 68.16 |

We further explore whether WRAP++ data can be integrated into a realistic mid-training pipeline without harming general capabilities.
We augment OLMo-3’s 100B-token mid-training mixture with 6B tokens of WRAP++ data and train for the full schedule. As a baseline, we train on the original OLMo-3 mid-training mixture under identical conditions.
Following niklaus2026finephrase, we evaluate both SimpleQA and the average performance across 12 general tasks (detailed in Appendix [F](#A6 "Appendix F Mid-Training Evaluation Benchmarks ‣ WRAP++: Web Discovery Amplified Pretraining")).
Table [4](#S4.T4 "Table 4 ‣ 4.6 Integration with OLMo-3 Mid-Training Data ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") shows that adding WRAP++ data yields a meaningful improvement on SimpleQA (+2.9 points pass@128) while maintaining a comparable general-benchmark average (68.16 vs. 68.24).
Notably, WRAP++ explicitly enhances knowledge-intensive tasks, yielding clear gains on MMLU Redux (+1.28, detailed in Appendix [F](#A6 "Appendix F Mid-Training Evaluation Benchmarks ‣ WRAP++: Web Discovery Amplified Pretraining")).
This demonstrates that WRAP++ integrates cleanly into full-scale mid-training, preserving broad capabilities while explicitly strengthening the model’s general knowledge foundation. Moreover, as the mid-training budgets of leading models grow toward the trillion-token regime (yang2025qwen3; zeng2026glm), the 80B-token scale of WRAP++ suggests strong potential for integration into future large-scale training pipelines.

## 5 Related Work

#### Synthetic Data for LLM Pretraining.

WRAP (maini2024wrap) established synthetic rephrasing as a practical pretraining paradigm, showing that rewriting web documents into cleaner QA-style text with instruction-tuned models can accelerate pretraining by ∼\sim3×\times. Subsequent work has expanded this design space along three main axes. First, on *rephrasing strategy*, Nemotron-CC (su-etal-2025-nemotron) extracts QA pairs and knowledge lists, REWIRE (nguyen2025recycling) introduces guided rewriting with explicit quality criteria, and later work explores additional target formats such as tutorials, FAQs, and mathematical reformulations (maini2025beyondweb; niklaus2026finephrase). Second, on *generator model*, studies spanning models from 270M to 27B parameters suggest that moderate-scale models (∼\sim1B–4B) already produce rephrasings competitive with much larger generators (maini2024wrap; niklaus2026finephrase). Third, on *source data quality*, rephrasing can upcycle low-quality web text (nguyen2025recycling), although higher-quality source documents still tend to yield stronger downstream performance (niklaus2026finephrase). A cross-cutting question concerns how synthetic and original data should be combined, since synthetic-only training often improves factual recall at the expense of broader capabilities, making mixture design important in practice (maini2024wrap; niklaus2026finephrase). Despite this progress, existing methods all synthesize from *single documents in isolation*. WRAP++ differs from this entire line of work by introducing cross-document knowledge discovery and amplification: instead of rewriting one document at a time, it discovers relational structure from web topology and jointly synthesizes training examples from related entity pairs, explicitly modeling relational knowledge that prior single-document approaches leave untapped.

## 6 Conclusion

We presented WRAP++, a framework that *amplifies* the associative context of factual knowledge by *discovering* cross-document relationships from web topology and synthesizing joint QA over related entity pairs. By mining relational motifs (dual-links and co-mentions) from Wikipedia hyperlinks, WRAP++ creates training data with richer relational structure and more diverse retrieval paths than single-document rewriting. On SimpleQA, WRAP++ substantially outperforms single-document approaches at 7B and 32B scales, with a favorable scaling trajectory up to 80B tokens.

## Appendix A Limitation

Our experiments instantiate WRAP++ on Wikipedia, which is a clean and entity-centric corpus. Extending to noisier web corpora (e.g., Common Crawl), where hyperlinks include advertisements, navigation elements, and low-quality references, will require additional filtering heuristics; we are actively exploring this direction.

## Appendix B Training Details

#### Architecture.

We use the OLMo-3 architecture (olmo2025olmo3), a decoder-only transformer with the Dolma-2 tokenizer (vocabulary size padded to a multiple of 128).
Experiments are conducted at two scales: OLMo-3-7B (7 billion parameters) and OLMo-3-32B (32 billion parameters). We utilize the olmo-core training framework (olmo20242olmo2furious), which provides a highly optimized and reproducible infrastructure for large-scale distributed training.
All models use FlashAttention-2 (dao2023flashattention2) as the attention backend.

#### Continued Pretraining.

All experiments initialize from official OLMo-3 pretrained checkpoints: step 1,413,814 for 7B and step 679,000 for 32B.
We load only the model weights and optimizer state (no trainer state) and continue pretraining on synthetic data mixtures for 1 epoch.
Table [5](#A2.T5 "Table 5 ‣ Continued Pretraining. ‣ Appendix B Training Details ‣ WRAP++: Web Discovery Amplified Pretraining") summarizes the hyperparameters for each configuration.

Table 5: Continued pretraining hyperparameters. All experiments use a linear decay schedule (no warmup, decaying to 0) and SkipStepAdamW (olmo2025olmo3) with β1=0.9\beta\_{1}\!=\!0.9, β2=0.95\beta\_{2}\!=\!0.95, weight decay 0.10.1 (embedding weights excluded), max gradient norm 1.01.0, and auxiliary z-loss with multiplier 10−510^{-5}. Parameters are stored in bfloat16 and gradients are reduced in float32 via HSDP.

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | OLMo-3-7B (8B) | OLMo-3-7B (80B) | OLMo-3-32B |
| Sequence length | 8,192 | 8,192 | 8,192 |
| Global batch size (tokens) | ∼\sim2M (2212^{21}) | ∼\sim2M (2212^{21}) | ∼\sim4M (4×2204\times 2^{20}) |
| Peak learning rate | 2.07×10−52.07\times 10^{-5} | 2.07×10−52.07\times 10^{-5} | 2.07×10−52.07\times 10^{-5} |
| LR schedule | Linear →\rightarrow 0 | Linear →\rightarrow 0 | Linear →\rightarrow 0 |
| Warmup steps | 0 | 0 | 0 |
| Training steps | 4,000 | 40,000 | 10,000 |
| Training tokens | ∼\sim8B | ∼\sim80B | ∼\sim80B |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Data parallel | HSDP (block wrap) | HSDP (block wrap) | HSDP (full wrap, shard 64) |
| Activation ckpt. | FFN-only | FFN-only | Budget (50%) |
| GPUs | 256 ×\times H20 | 256 ×\times H20 | 1024 ×\times H20 |

The learning rate of 2.07×10−52.07\times 10^{-5} is inherited from the official OLMo-3 mid-training recipe. We use zero warmup steps because the optimizer state is loaded from the pretrained checkpoint, ensuring stable training from the first step. For the OLMo-3-7B scale, training on ∼\sim8B tokens using 256 H20 GPUs takes approximately 4.5 hours with a Model FLOPs Utilization (MFU) of ∼\sim65%. For the larger OLMo-3-32B scale, training on ∼\sim80B tokens using 1,024 H20 GPUs requires approximately 1 day and 20 hours. These benchmarks demonstrate the efficiency and scalability of our training pipeline on modern hardware.

#### Data Format.

All synthetic QA data is formatted as plain text with “Question:” and “Answer:” delimiters, consistent with prior WRAP work.
For cross-document QA, the synthesized output directly states facts without referencing source passages, ensuring the model internalizes them as parametric knowledge rather than reading comprehension signals.

## Appendix C Evaluation Details

#### In-Context Learning Setup.

Because our continued pretraining experiments operate on base models (OLMo-3) that have not undergone instruction tuning, these models cannot reliably follow zero-shot formatting directives. To accurately probe their parametric knowledge, we adopt a 5-shot in-context learning protocol for all evaluations, including both SimpleQA and the 12 general benchmarks. Specifically, for each evaluation instance, we prepend the prompt with five demonstration question-answer pairs. For SimpleQA, these demonstrations are sampled directly from the SimpleQA dataset; to ensure strict evaluation integrity and prevent data contamination, any examples used as few-shot demonstrations are explicitly excluded from the active evaluation set during inference.

#### Decoding and Sampling Parameters.

We adopt distinct decoding strategies tailored to the nature of each benchmark. For SimpleQA, to compute the pass@kk metric (where n=128n=128), we employ nucleus sampling with a temperature of 0.60.6 and top-pp of 0.950.95 to provide a diverse distribution for knowledge recoverability analysis. In contrast, for the 12 general benchmarks, we use greedy decoding (temperature 0.00.0) to ensure deterministic and reproducible outputs across all model comparisons. These parameters are held constant across all model scales and data recipes to ensure a fair evaluation.

#### Unbiased Estimation of pass@kk.

While the pass@kk metric intuitively represents the probability of generating at least one correct answer within kk attempts, empirically estimating this by drawing exactly kk samples yields high variance. To achieve a more stable and unbiased estimate, we adopt the methodology introduced by chen2021codex. For each evaluation instance, we generate nn total samples (n≥kn\geq k) and determine the number of correct responses, cc. The unbiased estimator for pass@kk is then computed as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | pass​@​k=1−(n−ck)(nk)\text{pass}@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}} |  | (4) |

where (⋅⋅)\binom{\cdot}{\cdot} denotes the binomial coefficient. In our experiments, we generate n=128n=128 samples per question. This formulation efficiently leverages all nn generated samples to calculate the expected pass rate for any evaluation budget k≤nk\leq n, thereby reducing variance without requiring repeated sampling passes.

## Appendix D Synthesis Prompt Templates

#### Single-Document WRAP Prompt (Baseline).

Following maini2024wrap, we use the standard QA-style prompt for single-document synthesis:

> Convert the following paragraph into a conversational format with multiple tags of ‘‘Question:’’ followed by ‘‘Answer:’’.

#### WRAP++ Cross-Document Joint QA Prompt.

For cross-document synthesis, we provide the full text of both related entities and apply the following instruction. The prompt enforces three critical constraints: cross-document dependency, explicit factual chaining, and omniscient internalization (see §[2.3](#S2.SS3 "2.3 Cross-Document Joint QA Synthesis ‣ 2 Method: WRAP++ ‣ WRAP++: Web Discovery Amplified Pretraining")).

> You are an expert data generator for language model pretraining.
>
> Below are two related Wikipedia passages:
>
> [Passage A]
>   
> {text\_a}
>
> [Passage B]
>   
> {text\_b}
>
> Task:
>   
> 1) Generate high-quality synthetic QA pairs that REQUIRE information from BOTH Passage A and Passage B to answer.
>   
> 2) The Answer MUST begin with a step-by-step reasoning process. This reasoning must explicitly bridge facts from both passages.
>   
> 3) Do not use external knowledge.
>   
> 4) CRITICAL CONSTRAINT: The generated QA pair will be used to train a model WITHOUT these passages provided as context. Therefore, you MUST act as an omniscient AI stating absolute facts from your own inherent knowledge.
>   
>  - DO NOT use any attribution phrases like ‘According to Passage A’, ‘Passage B mentions’, ‘As stated in the text’, or ‘Based on the provided documents’.
>   
>  - State the facts directly and confidently.
>
> Output format (strict):
>   
> Question: [Insert a complex question that bridges facts from both passages]
>   
> Answer: [Acting as an omniscient AI, directly state all necessary factual premises from both passages, and logically synthesize them to derive the conclusion.]
>   
> Therefore, [State the final, concise answer.]

#### Synthesis Hyperparameters.

All QA synthesis uses Qwen3-30B-A3B-Instruct (FP8) as the generator with temperature 0.70.7, top-pp = 0.80.8, and a maximum output length of 32,768 tokens.
For the synthesis model scale ablation (§[4.4](#S4.SS4 "4.4 Effect of Synthesis Model Scale ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining")), we additionally use Qwen3-235B-A22B-Instruct.

## Appendix E Relation Discovery Statistics

Table 6: Statistics of the hyperlink relations discovered from FineWiki (English Wikipedia).

|  |  |
| --- | --- |
| Statistic | Value |
| Source corpus | FineWiki (English) |
| Raw corpus tokens | ∼\sim8.4B |
| Wikipedia articles processed | ∼\sim6.7M |
| Dual-link pairs (A↔BA\leftrightarrow B) | ∼\sim9.6M |
| Co-mention pairs (A→E←BA\rightarrow E\leftarrow B, A→BA\rightarrow B) | ∼\sim232M |
| Dual-link synthesized tokens | ∼\sim3B |
| Co-mention synthesized tokens | ∼\sim79.7B |
| Total WRAP++ tokens | ∼\sim82.7B |
| Single-doc WRAP tokens (baseline) | ∼\sim5.4B |
| Extended WRAP tokens (baseline) | ∼\sim17.4B |

Table [6](#A5.T6 "Table 6 ‣ Appendix E Relation Discovery Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") reports the relation discovery and synthesis statistics.
The dual-link motif yields a relatively small but high-precision set of ∼\sim9.6M entity pairs, while the co-mention motif provides a much larger pool of ∼\sim232M pairs, enabling substantial combinatorial expansion.
The total synthesized corpus of ∼\sim82.7B tokens is approximately 15×15\times larger than single-document WRAP on the same source, illustrating the amplification advantage of relation-driven cross-document synthesis.

## Appendix F Mid-Training Evaluation Benchmarks

The 12 general tasks used for evaluation in Section [4.6](#S4.SS6 "4.6 Integration with OLMo-3 Mid-Training Data ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining") are categorized as follows:

* •

  General Knowledge: ARC (clark2018think), MMLU Redux (gema2024are)
* •

  Reading Comprehension: SQuAD v2 (rajpurkar2018know), DROP (dua2019drop)
* •

  Reasoning: OpenBookQA (mihaylov2018can), CSQA (talmor2019commonsenseqaquestionansweringchallenge)
* •

  Natural Language Understanding: WinoGrande (sakaguchi2021winogrande), PIQA (bisk2020piqa), HellaSwag (zellers2019hellaswag)
* •

  Math: GSM8K (cobbe2021training)
* •

  Table Understanding: WikiTableQuestions (pasupat2015compositional), TriviaQA (joshi2017triviaqa)

Table [7](#A6.T7 "Table 7 ‣ Appendix F Mid-Training Evaluation Benchmarks ‣ WRAP++: Web Discovery Amplified Pretraining") provides per-benchmark results for the mid-training integration experiment described in Section [4.6](#S4.SS6 "4.6 Integration with OLMo-3 Mid-Training Data ‣ 4 Ablations and Analysis ‣ WRAP++: Web Discovery Amplified Pretraining").

Table 7: Per-benchmark breakdown for mid-training integration (OLMo-3-7B, full 100B-token schedule, 3-shot cloze format). Accuracy (%) is reported for all tasks except SQuAD v2 and DROP, which use token-level F1.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Category | Benchmark | Base | Midtrain | WRAP++ Mix |
| General Knowledge | ARC | 77.47 | 84.98 | 85.07 |
| MMLU Redux | 60.30 | 65.30 | 66.58 |
| Reading Comprehension | SQuAD v2 | 42.27 | 48.59 | 49.99 |
| DROP | 40.51 | 66.51 | 65.36 |
| Reasoning | OpenBookQA | 76.40 | 85.20 | 82.40 |
| CSQA | 72.15 | 74.20 | 72.97 |
| Language Understanding | WinoGrande | 53.28 | 65.11 | 62.98 |
| PIQA | 73.99 | 74.54 | 74.92 |
| HellaSwag | 59.83 | 70.15 | 73.12 |
| Math | GSM8K | 38.36 | 79.53 | 77.79 |
| Table & Trivia | WikiTableQ | 36.55 | 43.74 | 44.46 |
| TriviaQA | 62.42 | 61.01 | 62.23 |
| Average | | 57.79 | 68.24 | 68.16 |

![Refer to caption](/html/2604.06829/assets/x4.png)


Figure 4: Per-benchmark score trajectories during mid-training with the WRAP++ Mix (100B tokens) on OLMo-3-7B. Each subplot tracks one of the 12 evaluation tasks over the course of training. All benchmarks exhibit a clear upward trend, with knowledge-intensive tasks (e.g., HellaSwag, GSM8K, CSQA) showing the most pronounced gains.

## Appendix G Additional Ablation: Co-Mention with Three Documents

In the main co-mention motif (A→E←BA\rightarrow E\leftarrow B while A→BA\rightarrow B), we use the two target entities AA and BB as input to joint QA synthesis.
A natural extension is to also include the bridging page EE as a third input document, potentially providing additional shared context.
We use a dedicated 3-document prompt template that instructs the synthesis model to generate QA requiring facts from all three passages.

Table 8: Co-mention synthesis using 2 vs. 3 input documents (OLMo-3-7B, ∼\sim8B tokens, SimpleQA pass@8).

| Co-mention Variant | SimpleQA pass@8 |
| --- | --- |
| 2-doc (entities AA, BB only) | 15.74 |
| 3-doc (entities AA, BB + bridge EE) | 15.42 |

Surprisingly, including the bridging page EE does not improve—and slightly hurts—performance (Table [8](#A7.T8 "Table 8 ‣ Appendix G Additional Ablation: Co-Mention with Three Documents ‣ WRAP++: Web Discovery Amplified Pretraining")).
We hypothesize that the bridging page introduces distracting context: since EE typically links to many entities, its content is broad and may divert the synthesis model from focusing on the specific relationship between AA and BB.
The 2-document formulation used in WRAP++ strikes a better balance between relational grounding and synthesis focus.

## Appendix H Synthesized QA Data Statistics

We report comprehensive statistics of the synthesized WRAP++ dataset to characterize the length distributions and data composition. Statistics are computed over the full corpus of 240,658,065 QA instances across 24,224 JSONL files, processed in parallel using 256 workers. Character-level and word-level lengths are measured on the raw synthesized text after extracting the “Question:” and “Answer:” fields from each record.

#### Dataset Composition.

Table [9](#A8.T9 "Table 9 ‣ Dataset Composition. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") summarizes the overall dataset composition by relation type.

Table 9: Composition of the WRAP++ synthesized QA dataset by relation type.

| Relation Type | QA Instances | Proportion |
| --- | --- | --- |
| Co-mention | 231,292,954 | 96.1% |
| Dual-link | 9,365,111 | 3.9% |
| Total | 240,658,065 | 100.0% |

#### Length Distributions.

Table [10](#A8.T10 "Table 10 ‣ Length Distributions. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") reports the distributional statistics of the synthesized QA text. Questions are concise (median 203 characters, 32 words), while answers are substantially longer (median 1,386 characters, 212 words), reflecting the explicit factual chaining constraint that requires step-by-step reasoning before stating the final conclusion. The overall QA length is concentrated in the 1,000–2,000 character range (56.1% of all instances), with 27.6% in the 2,000–5,000 range and 16.0% in the 500–1,000 range. Fewer than 0.3% of instances fall outside the 500–5,000 character window, indicating a well-controlled generation process. Figure [5](#A8.F5 "Figure 5 ‣ Length Distributions. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") visualizes the question and answer length distributions at both character and word levels.

![Refer to caption](/html/2604.06829/assets/x5.png)


Figure 5: Question vs. answer length distributions at character level (left) and word level (right). Questions are tightly concentrated around a median of 203 characters (32 words), while answers exhibit a broader, right-skewed distribution with a median of 1,386 characters (212 words), reflecting the explicit factual chaining required by the synthesis prompt.




Table 10: Length statistics of WRAP++ synthesized QA data (aggregated over all 240.7M instances). “Chars” denotes character count; “Words” denotes whitespace-delimited token count.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Field | Mean | Std | Min | P5 | P25 | Median | P75 | P95 |
| Character-level | | | | | | | | |
| Question | 207 | 43 | 45 | 143 | 177 | 203 | 233 | 283 |
| Answer | 1,424 | 624 | 139 | 563 | 938 | 1,386 | 1,817 | 2,463 |
| QA (combined) | 1,651 | 637 | 242 | 766 | 1,161 | 1,610 | 2,051 | 2,716 |
| Word-level | | | | | | | | |
| Question | 33 | 7 | 7 | 23 | 28 | 32 | 37 | 45 |
| Answer | 217 | 91 | 16 | 92 | 149 | 212 | 272 | 364 |

#### Comparison Across Relation Types.

Table [11](#A8.T11 "Table 11 ‣ Comparison Across Relation Types. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") compares the QA length characteristics between the two relation motifs. Dual-link instances produce slightly shorter answers (median 1,210 vs. 1,394 characters), likely because mutual references tend to encode more focused bilateral relationships, whereas co-mention pairs often involve broader categorical or analogical connections that require more elaboration. Despite these differences, both subsets maintain similar question lengths and overall distributional shape. Figure [6](#A8.F6 "Figure 6 ‣ Comparison Across Relation Types. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") overlays the QA length histograms for both relation types, and Figure [7](#A8.F7 "Figure 7 ‣ Comparison Across Relation Types. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") provides a violin plot comparison across three length dimensions.

![Refer to caption](/html/2604.06829/assets/x6.png)


Figure 6: Synthesized QA length distribution by relation type. Both subsets exhibit a similar unimodal shape, with the co-mention distribution (blue) shifted slightly rightward relative to dual-link (red), consistent with the longer factual chains required to bridge co-mentioned entities.




Table 11: Median QA lengths (characters) by relation type. Word-level medians are shown in parentheses.

| Relation Type | Question | Answer | QA Total |
| --- | --- | --- | --- |
| Co-mention | 204 (32 words) | 1,394 (213 words) | 1,618 |
| Dual-link | 197 (32 words) | 1,210 (189 words) | 1,428 |
| Overall | 203 (32 words) | 1,386 (212 words) | 1,610 |

![Refer to caption](/html/2604.06829/assets/x7.png)


Figure 7: Violin plot comparison of question length, answer length, and answer word count between co-mention and dual-link subsets. Black horizontal lines indicate medians. Both motifs produce similarly distributed questions, while co-mention answers are moderately longer, reflecting the additional elaboration needed to bridge indirectly related entities.

#### Source Document Lengths.

The input Wikipedia passages exhibit substantial length variation. The first passage (text\_a) has a median length of 4,458 characters (P5–P95: 519–27,963), while the second passage (text\_b) is generally longer with a median of 9,575 characters (P5–P95: 943–50,004, where 50,004 indicates truncation at the maximum context window). This asymmetry arises because co-mention pairs order documents by the directed edge A→BA\rightarrow B, where BB (the referenced entity) tends to be a more prominent article. Figure [8](#A8.F8 "Figure 8 ‣ Source Document Lengths. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") visualizes this distributional asymmetry.

![Refer to caption](/html/2604.06829/assets/x8.png)


Figure 8: Source document length distributions for the two input passages. Passage A (the referencing entity) is typically shorter (median ≈\approx4,700 chars), while Passage B (the referenced entity) tends to be longer and more prominent (median ≈\approx9,500 chars), with a visible mass accumulation at the 50K truncation boundary.

#### QA Length Bucketed Distribution.

Table [12](#A8.T12 "Table 12 ‣ QA Length Bucketed Distribution. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") and Figure [9](#A8.F9 "Figure 9 ‣ QA Length Bucketed Distribution. ‣ Appendix H Synthesized QA Data Statistics ‣ WRAP++: Web Discovery Amplified Pretraining") provide a bucketed view of the combined QA and answer length distributions. The synthesis process produces a unimodal distribution with the majority of instances in the 1,000–2,000 character range. No instances have empty answers, and fewer than 0.01% of answers are shorter than 200 characters, confirming that the explicit factual chaining constraint effectively prevents degenerate outputs.

![Refer to caption](/html/2604.06829/assets/x9.png)


Figure 9: Bucketed length distributions computed over the full dataset of 240.7M instances. Left: combined QA length; right: answer-only length. The 1K–2K character bucket dominates both distributions (>>55%), with a secondary concentration in the 2K–5K range for QA and the 500–1K range for answers. Extreme lengths (>>5K or <<200) are negligible.




Table 12: Bucketed length distributions for the combined QA text and answer text (character-level).

| Length Range (chars) | QA (combined) | | Answer only | |
| --- | --- | --- | --- | --- |
| Count | % | Count | % |
| [0,200)[0,200) | 0 | 0.0 | 2,479 | <0.01{<}0.01 |
| [200,500)[200,500) | 456,399 | 0.2 | 6,771,177 | 2.8 |
| [500,1,000)[500,1{,}000) | 38,609,776 | 16.0 | 61,411,377 | 25.5 |
| [1,000,2,000)[1{,}000,2{,}000) | 135,127,249 | 56.1 | 132,242,235 | 55.0 |
| [2,000,5,000)[2{,}000,5{,}000) | 66,336,557 | 27.6 | 40,114,788 | 16.7 |
| [5,000,10,000)[5{,}000,10{,}000) | 99,207 | <0.1{<}0.1 | 89,225 | <0.1{<}0.1 |
| [10,000,∞)[10{,}000,\infty) | 28,877 | <0.1{<}0.1 | 26,784 | <0.1{<}0.1 |

## Appendix I Qualitative Analysis

#### Example: WRAP vs. WRAP++ Synthesis.

Source entity A: Ludwig Göransson (Swedish composer, film score artist)
  
Source entity B: Oppenheimer (2023 film directed by Christopher Nolan)

Table 13: Qualitative comparison between single-document WRAP and cross-document WRAP++ synthesis for the Ludwig Göransson and *Oppenheimer* example.

|  |  |
| --- | --- |
| WRAP (single-document QA on entity A) | WRAP++ (cross-document QA on entities A+B) |
| Q: Who is Ludwig Göransson? A: Ludwig Göransson is a Swedish composer known for film scores. Q: What films has Göransson scored? A: He has scored *Black Panther*, *Tenet*, and *Oppenheimer*. | Q: What films earned Nolan’s original Tenet composer and his replacement their second Oscars? A: Hans Zimmer turned down Tenet for Dune, which earned his second Oscar (after The Lion King). His replacement, Ludwig Göransson, later won his second Oscar for Oppenheimer (after Black Panther). Therefore, the answer is Dune and Oppenheimer. |

As shown in Table [13](#A9.T13 "Table 13 ‣ Example: WRAP vs. WRAP++ Synthesis. ‣ Appendix I Qualitative Analysis ‣ WRAP++: Web Discovery Amplified Pretraining"), the WRAP++ output creates richer associative context: it contrasts with a commonly confused entity (Zimmer), provides cross-film comparisons, and generates reverse-direction queries—all contributing to more robust knowledge encoding.

[◄](/html/2604.06828)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.06829)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.06829)
[View original  
on arXiv](https://arxiv.org/abs/2604.06829)[►](/html/2604.06830)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed May 6 01:32:34 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

var canMathML = typeof(MathMLElement) == "function";
if (!canMathML) {
var body = document.querySelector("body");
body.firstElementChild.setAttribute('style', 'opacity: 0;');
var loading = document.createElement("div");
loading.setAttribute("id", "mathjax-loading-spinner");
var message = document.createElement("div");
message.setAttribute("id", "mathjax-loading-message");
message.innerText = "Typesetting Equations...";
body.prepend(loading);
body.prepend(message);
var el = document.createElement("script");
el.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
document.querySelector("head").appendChild(el);
window.MathJax = {
startup: {
pageReady: () => {
return MathJax.startup.defaultPageReady().then(() => {
body.removeChild(loading);
body.removeChild(message);
body.firstElementChild.removeAttribute('style');
}); } } };
}

// Auxiliary function, building the preview feature when
// an inline citation is clicked
function clicked\_cite(e) {
e.preventDefault();
let cite = this.closest('.ltx\_cite');
let next = cite.nextSibling;
if (next && next.nodeType == Node.ELEMENT\_NODE && next.getAttribute('class') == "ar5iv-bibitem-preview") {
next.remove();
return; }
// Before adding a preview modal,
// cleanup older previews, in case they're still open
document.querySelectorAll('span.ar5iv-bibitem-preview').forEach(function(node) {
node.remove();
})
// Create the preview
preview = document.createElement('span');
preview.setAttribute('class','ar5iv-bibitem-preview');
let target = document.getElementById(this.getAttribute('href').slice(1));
target.childNodes.forEach(function (child) {
preview.append(child.cloneNode(true));
});
let close\_x = document.createElement('button');
close\_x.setAttribute("aria-label","Close modal for bibliography item preview");
close\_x.textContent = "×";
close\_x.setAttribute('class', 'ar5iv-button-close-preview');
close\_x.setAttribute('onclick','this.parentNode.remove()');
preview.append(close\_x);
preview.querySelectorAll('.ltx\_tag\_bibitem').forEach(function(node) {
node.remove();
});
cite.parentNode.insertBefore(preview, cite.nextSibling);
return;
}
// Global Document initialization:
// - assign the preview feature to all inline citation links
document.querySelectorAll(".ltx\_cite .ltx\_ref").forEach(function (link) {
link.addEventListener("click", clicked\_cite);
});
