---
arxiv: '2602.12413'
authors:
- Ari Spiesberger
- Juan J. Vazquez
- Nicky Pochinkov
- Tomáš Gavenčiak
- Peli Grietzer
- Gavin Leech
- Nandi Schoots
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Soft Contamination Means Benchmarks Test Shallow Generalization
url: https://arxiv.org/abs/2602.12413
year: 2026
---

# Soft Contamination Means Benchmarks Test Shallow Generalization

Ari Spiesberger
  
Juan J Vazquez
  
Nicky Pochinkov
  
Tomáš Gavenčiak
  
Peli Grietzer
  
Gavin Leech
  
Nandi Schoots

###### Abstract

If LLM training data is polluted with benchmark test data, then benchmark performance gives biased estimates of out-of-distribution (OOD) generalization.
Typical ‘decontamination’ filters use nn-gram matching which fail to detect ‘semantic’ duplicates: sentences with equivalent (or near-equivalent) content that are not close in string space.
We study this ‘soft’ contamination of training data by semantic duplicates. Among other experiments, we embed the Olmo3 training corpus and find that:
1) contamination remains widespread, e.g. we find semantic duplicates for 78% of CodeForces and exact duplicates for 50% of ZebraLogic problems;
2) including semantic duplicates of benchmark data in training does improve benchmark performance; and
3) when finetuning on duplicates of benchmark datapoints, performance also improves on truly-held-out datapoints from the same benchmark.
We argue that recent benchmark gains are thus confounded: the prevalence of soft contamination means gains reflect both genuine capability improvements and the accumulation of test data and effective test data in growing training corpora.

Machine Learning, ICML

## 1 Introduction

LLM scores on hard reasoning (incl. coding) benchmarks have been growing rapidly, with many benchmarks nearing saturation even for smaller, open-source models (Edelman and Lee, [2025](#bib.bib12 "AI capabilities progress has sped up"); Maslej et al., [2025](#bib.bib11 "Artificial intelligence index report 2025")). Does this trend purely reflect growth in LLMs’ general, OOD reasoning capability, or does it also reflect limitations of the benchmarking procedure? We address this question by combining existing data-contamination detection methods with novel finetuning experiments to diagnose what we call shallow generalization on benchmarks: benchmark-specific performance gains from training on datapoints that are qualitatively typical of the benchmark. Using the open-data model Olmo3 as a case study, we show that modern LLM training corpora include data that qualitatively function like samples from major reasoning benchmarks, leading to benchmark scores that to some extent demonstrate shallow generalization rather than general reasoning capability. We thus hypothesize that the rapid increase in LLMs’ performance on reasoning benchmarks partly reflects the rapid growth of LLMs’ corpora size and downstream shallow generalization that tunes models to individual benchmarks via sample-like data. If true, then recent progress on major reasoning benchmarks is weaker evidence for the true pace of AI progress (conceived as OOD generalization).

Table 1: Comparison to past literature on data contamination, focusing on semantic-duplicates studies. We compare whether past work studies semantic duplicates of test data; estimates contamination prevalence in training corpora; has methods to automatically screen semantic overlap; quantifies the effects of contamination on downstream performance; employs finetuning and whether the finetuning data are ecologically realistic; tests in-benchmark generalization (training on duplicates of some benchmark items improves performance on *other* held-out items from the same benchmark); and the scale of data, models, and benchmarks studied.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paper | Semantic  dupes? | Contam.  est.? | Auto  detect.? | Effect  est.? | With  FT? | FT data comp.  realistic? | In-bench  gen.? | Data  scale | Open data  attrib.? | Model  scale | Major  evals |
| Ours | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Large | ✓ | 7B | CodeForces (+ MBPP, MuSR, ZebraLogic) |
| Magar and Schwartz ([2022](#bib.bib18 "Data contamination: from memorization to exploitation")) | ✗ | ✗ | - | ✓ | ✓ | ✗ | ✗ | Small | ✓ | 0.1–0.3B | SST-5, SST-2, SNLI |
| Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | Medium | ✓ | 13B | MMLU, GSM8K, HumanEval |
| Riddell et al. ([2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models")) | ✓ | ✓ | ✓ | ✓ | ✗ | - | ✗ | Medium | ✓ | ≤\leq16B | MBPP, HumanEval |
| Shilov et al. ([2025](#bib.bib26 "The mosaic memory of large language models")) | ✗ | ✗ | - | ✓ | ✓ | ✗ | ✗ | Small | ✗ | ≤\leq2.7B | Canary memorization (MIA) |
| Xu et al. ([2025](#bib.bib21 "SSA: semantic contamination of LLM-driven fake news detection")) | ✓ | ✗ | ✓ | ✓ | ✗ | - | ✗ | Varied | ✗ | 0.5–72B | LIAR2 |

LLM training corpora have grown by a factor of at least 10,000x since 2020 (Epoch AI, [2025](#bib.bib10 "Data on ai models")). Thus, there is reason to expect more benchmark test-examples to be included in the training corpora of recent LLMs. While AI labs make good-faith efforts to remove syntactic duplicates of benchmark items from their corpora (OpenAI et al., [2024](#bib.bib4 "GPT-4 technical report"); Olmo et al., [2025](#bib.bib33 "Olmo 3"); Anthropic, [2025b](#bib.bib3 "System card: claude opus 4.5")), ‘softer’ forms of contamination are extremely hard to detect, and may well be the product of parallel evolution rather than the product of a data leak. Nevertheless, the presence of benchmark-convergent data in LLMs’ training corpora can act as a major confounder with regard to the type of generalization evidenced by benchmark scores. Reasoning (incl. coding) benchmark scores, in particular, are typically intended not as measures of LLMs’ within-distribution generalization capabilities (comparable to generalization from one subset of the benchmark to another subset), but as tests of the application of fundamental capabilities.

To estimate the significance of shallow generalization as a confounder in benchmark results, we gauge the prevalence of exact and semantic duplicates of items from major reasoning benchmarks in the training corpus of Olmo3. We then conduct finetuning experiments with exact duplicates, semantic duplicates, and close embedding neighbors to test their capacity to induce shallow generalization on a target reasoning-benchmark. While our experiments distinguish between different kinds of ‘shallow generalization’ gains – gains on a benchmark item from training on its semantic duplicates (Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")); Riddell et al. ([2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models"))); gains on a benchmark item from training on exact duplicates of other items in the benchmark; gains on a benchmark item from training on semantic duplicates of other items in the benchmark – our discussion frames them as a unified phenomenon from the viewpoint of AI-progress benchmarking: effects of corpus growth that don’t reduce to test-memorization but fall short of the capability-growth that benchmarks are designed to measure.

Some terminology: An exact duplicate of test data is an example in the training corpus which is syntactically identical (perhaps up to some number of nn-grams) to some item in a relevant test set. A semantic duplicate of test data is an example in the training corpus which has the same meaning (in some sense) as some item in a relevant test set (Riddell et al., [2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models")). We call contamination soft when it involves semantic duplicates. We call generalization shallow when it’s limited to a combination of within-distribution generalization and generalization across semantic duplicates.

Our contributions:

* •

  Large rates of contamination: We screen 1% of the pretraining data and all of the finetuning data of Olmo3 for semantic duplicates ‘in the wild’ by using their embedding distance to benchmark data, which is far more than previous studies have investigated. Despite decontamination efforts in the data preparation of Olmo3, we find much more contamination than previous studies found, likely because we investigate more data;
* •

  Shallow generalization: We finetune Olmo3 on duplicates of a subset of the (MuSR, ZebraLogic and MBPP) benchmark and find that benchmark performance also improves on unseen benchmark data. For some benchmarks we find that finetuning on semantic duplicates has the same effect size as finetuning on exact duplicates (an increase of 20% on both seen and unseen items), while finetuning on close embedding neighbors has no effect. Finetuning on one benchmark does not typically improve performance on related benchmarks, suggesting that the generalization in our finetuning experiments is strictly ‘shallow’;
* •

  Ecologically valid amounts of contamination have a substantial effect: We use our findings on the rate of semantic duplicates in the wild to design a finetuning experiment with an ecologically valid mix of benchmark-semantic-duplicate datapoints and clean datapoints in the finetuning corpus, and find that finetuning substantially improves benchmark performance (by around 15%).

## 2 Related Work

Contamination of LLM training corpora by benchmark test items is a well-trodden topic. Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") summarizes the key differences between our work and prior studies of benchmark contamination.

The first wave of data-contamination studies (Magar and Schwartz, [2022](#bib.bib18 "Data contamination: from memorization to exploitation"); Elazar et al., [2024](#bib.bib23 "What’s in my big data?"); Jiang et al., [2024](#bib.bib24 "Investigating data contamination for pre-training language models")) focused on the prevalence and impact of exact syntactic duplicates (i.e. word-for-word matches in string space), with later work (Zhou et al., [2025](#bib.bib25 "LessLeak-bench: a first investigation of data leakage in LLMs across 83 software engineering benchmarks"); Shilov et al., [2025](#bib.bib26 "The mosaic memory of large language models")) extending the analysis to partial syntactic duplicates. More recently, researchers have begun to study indirect or ‘semantic’ contamination (Yang et al., [2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples"); Riddell et al., [2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models"); Xu et al., [2025](#bib.bib21 "SSA: semantic contamination of LLM-driven fake news detection")), where data in the training corpora is equivalent to benchmark-items in substantive content without sharing n-gram sequences or other syntactical properties. Our work draws on techniques and best-practices from the semantic contamination literature, but aims at a partially different end: The literature on semantic contamination focuses on the relationship between performance on a benchmark item and training-exposure to that same item’s semantic duplicates, which it treats as a variant on memorization. Our work instead studies semantic duplicates as a source of shallow generalization phenomena, which include generalization from semantic duplicates to their benchmark-item equivalents but more importantly include within-benchmark-like generalization to non-duplicated items in the benchmark.

Studies of the prevalence of corpora-contamination by exact duplicates of benchmark data typically deploys two kinds of methods: searching for benchmark-item duplicates in open datasets (Elazar et al., [2024](#bib.bib23 "What’s in my big data?"); Riddell et al., [2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models"); Zhou et al., [2025](#bib.bib25 "LessLeak-bench: a first investigation of data leakage in LLMs across 83 software engineering benchmarks")), and memorization testing using ‘membership inference’ style techniques (Shi et al., [2023](#bib.bib28 "Detecting pretraining data from large language models")). When studying semantic-duplicates contamination, by contrast, memorization diagnostics are unlikely to capture the right contamination effects. Search in open datasets is therefore preferred in the (small) literature, using a heuristic semantic distance to guide search and human judgment, AI-assistant judgment, or plagiarism-detection software to assign ‘semantic duplicate’ status. Previous work by Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")) and Riddell et al. ([2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models")) has provided high-quality estimates of the prevalence of semantic duplicates of items from major reasoning benchmarks including HumanEval, MMLU, and GSK8k (Chen et al., [2021](#bib.bib29 "Evaluating large language models trained on code"); Hendrycks et al., [2020](#bib.bib37 "Measuring massive multitask language understanding"); Cobbe et al., [2021](#bib.bib38 "Training verifiers to solve math word problems")), in widely-used training corpora such as The Pile, StarCoderData, and RedPajama (Gao et al., [2020](#bib.bib30 "The pile: an 800gb dataset of diverse text for language modeling"); Li et al., [2023](#bib.bib31 "StarCoder: may the source be with you!"); Weber et al., [2024](#bib.bib32 "RedPajama: an open dataset for training large language models")).

We use a method convergent with that of Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")) to estimate the prevalence of semantic duplicates in Dolma (Soldaini et al., [2024](#bib.bib20 "Dolma: an open corpus of three trillion tokens for language model pretraining research")), Olmo’s custom training corpus. While our search covers a much larger dataset and finds many more semantic duplicates per benchmark item, our results are consistent with their findings in terms of the density of semantic duplicates in LLMs’ training corpora. (Note that because AI labs make ongoing decontamination efforts informed by the scientific literature, the prevalence of contaminating data of any type in SOTA training corpora is a potentially moving target and cannot be automatically assumed from older work.)

Studies attempting to estimate the effect of data-contamination (whether semantic or exact) on the integrity of benchmark scores have, to our knowledge, almost exclusively focused on memorization and memorization-like ‘item-to-item’ effects (one exception is Xu et al. ([2025](#bib.bib21 "SSA: semantic contamination of LLM-driven fake news detection")), which studies fake-news detection and employs a domain-specific concept of entity-exposure). Two central methods in the literature are finetuning on contaminated data to simulate training-exposure (assuming or verifying that the model had no or limited prior exposure), which Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")) applies to semantic duplicates, and using duplicate-prevalence data to test for correlations between a benchmark-item’s rate of duplication in a model’s training corpus and the model’s performance on the item, which Riddell et al. ([2024](#bib.bib19 "Quantifying contamination in evaluating code generation capabilities of language models")) applies to semantic duplicates. Our work uses a finetuning approach, but tests not only gains on a benchmark item from finetuning on its own semantic duplicates, but also gains on benchmark items from finetuning on semantic and on exact duplicates of other items in the benchmark. Inspired by Kocyigit et al. ([2025](#bib.bib22 "Overestimation in LLM evaluation: a controlled large-scale study on data contamination’s impact on machine translation"))’s study of the memorization-effects caused by injecting realistic dosages of exact duplicates into a clean finetuning corpus, we also design the first (to our knowledge) ecologically valid finetuning study of the effect of training-exposure to semantic duplicates.

## 3 Methodology

In order to know what data the studied model has seen and allow for an exhaustive scientific analysis, our experiments use Olmo-3-7b (Olmo et al., [2025](#bib.bib33 "Olmo 3")) a fully open-source (in particular, open-data) model. In addition, while some closed models allow for finetuning, with a closed corpus we cannot rule out there having been already trained on the examples (or neighbors of the examples) we finetune on, which would confound our effect estimates111Code at <https://github.com/AriSpiesberger/Soft-Contamination-Prevelance>.

### 3.1 Benchmarks

We select benchmarks based on: 1) the ability to generate new synthetic samples using an existing pipeline, 2) the tractability of creating semantic duplicates by modifying original samples, and 3) the likelihood of encountering semantic duplicates of benchmark samples in the wild.
We also prioritize benchmarks on which Olmo3 is not saturated, making it easier to track performance improvement or degradation during finetuning.

MBPP (Austin et al., [2021](#bib.bib17 "Program synthesis with large language models")).
Dataset of programming questions and validated solutions in Python, with test cases to check correctness. Since they are solvable by entry-level programmers, it is simple to generate correct alternative python solutions or translations in other programming languages.
We expected that the training corpora would contain semantic duplicates of MBPP test data. Unlike for other benchmarks, LLM-assisted annotation to detect such duplicates may be feasible without requiring very complex reasoning, but would still be undetectable by typical deduplication methods.
We use the sanitized test set which contains 257 tasks.

CodeForces (Penedo et al., [2025](#bib.bib35 "CodeForces")).
Dataset of competitive programming tasks with larger text inputs and problem context than the average code benchmark, such as MBPP. We expected equivalent tasks with less context to appear in training corpora which would be difficult to identify without manual (or LLM) annotation.
We use the 468 problems in the default test set in our experiments.

MuSR (Sprague et al., [2024](#bib.bib1 "MuSR: testing the limits of chain-of-thought with multistep soft reasoning")).
Dataset used to evaluate multi-step reasoning involving long narratives generated from a ground-truth logic tree built algorithmically. These trees are provided for each original sample by the authors, which allows the creation of semantic duplicates by regenerating story context from the same tree, and slightly different narratives by modifying non-critical tree branches. We focus on the ‘murder mysteries’ task with 250 problems.

ZebraLogic (Lin et al., [2025](#bib.bib16 "ZebraLogic: on the scaling limits of llms for logical reasoning")).
Dataset of 1000 logic grid puzzles of varying complexity levels used to evaluate logical reasoning capabilities.
We chose ZebraLogic because Olmo3 Instruct is not saturated on it (having 32.9% accuracy).
Additionally, as it is a non-coding benchmark it increases the variety of our suite of benchmarks.

### 3.2 Finding Semantic Duplicates in the Wild

Olmo3 Corpus Data.
We embed 1% of the Olmo3 Base training data (Dolma3 and Dolmino) and all of the Olmo3 Instruct finetuning data (Dolci SFT, Dolci Instruct DPO and Dolci Instruct RL).
To sample from the base training data, we employ a stratified reservoir sampling strategy that preserves the corpus’s hierarchical structure (e.g., common\_crawl/art-and-science).
As data is ingested, it is parsed into chunks and routed into distinct reservoirs corresponding to each sub-source.
We then construct the final dataset by drawing from these reservoirs in exact proportion to their original volume, ensuring the sample’s distribution remains consistent with the full corpus topology.
See Appendix [A.1](#A1.SS1 "A.1 Olmo3 Instruct Training Datasets ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") for more details on these datasets.

Embeddings.
We used llama-embed-nemotron-8b (Babakhin et al., [2025](#bib.bib41 "Llama-embed-nemotron-8b: a universal text embedding model for multilingual and cross-lingual tasks")) to embed the above datasets. At time of writing this model is number 2 on the Massive Text Embedding Benchmark (MTEB) leaderboard (Muennighoff et al., [2023](#bib.bib40 "Mteb: massive text embedding benchmark")).
All embeddings were done in FP16 precision.

Cosine Similarity.
We embed both Olmo3 corpus data and benchmark data, and calculate the cosine similarity between data points.
The benchmark data comparisons consist primarily of the MBPP and CodeForces datasets.
When comparing with pretraining data sets, we split MBPP into inputs and outputs.
When comparing MBPP with instruct data, we join MBPP inputs and outputs to match the prompt response format in the instruct data.
In Appendix [A.2](#A1.SS2 "A.2 Extended Cosine Similarity discussion ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we also present cosine similarity matches between corpus data and MuSR and ZebraLogic.
Our comparison consists of taking the cosine similarity of the benchmark point embedding with the corpus text embeddings.

Investigating Semantic and Exact Duplicate Status of High Cosine Similarity Matches

For MuSR the highest cosine similarity matches are around 0.400.40, and upon manual inspection it appears there are no duplicates, or MuSR-like problems in the training corpora. We conclude that the Olmo3 datasets have no semantic duplicates of MuSR.
In the case of ZebraLogic we found many exact duplicates and a few semantic duplicates. Additionally to some semantic duplicates, most top cosine similarity matches were Einstein riddles and Zebra puzzles available online, but do not match in complexity or sample integrity to the dataset samples.
After manual inspection of the top matches for MBPP and CodeForces we decided to sample, and annotate them using an LLM.

*Exact Duplicates: Sampling and Annotating ZebraLogic.*
We observed that several ZebraLogic tasks were present in the training corpora verbatim. So we ran an annotation experiment to get a rate of exact duplicates. This consisted in checking the 10 highest cosine similarity matches for each test point (adding up to 10,000 annotations for the entire benchmark).

*Semantic Duplicates: Sampling and Annotating MBPP and CodeForces.*
We investigate matches between MBPP and CodeForces for each of the five training datasets.
Per dataset and benchmark, we select the points in the top 0.1% similarity.
From these we either take the top 100, or randomly sample 100 points, and use a diffused model of gemini-3-flash-preview (Google DeepMind, [2025](#bib.bib42 "Gemini 3 flash model card")) to categorize whether the two texts are semantic duplicates or not.
Obtaining labels for whether the corpus text is a semantic duplicate or not, the type of semantic duplicate (exact, equivalent, subset, superset), reasoning of the choice, and the confidence of the label. See more details on the annotation pipeline in Appendix [A.4](#A1.SS4 "A.4 Annotation schemes for high cosine similarity matches ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").

### 3.3 Generating Synthetic Semantic Duplicates

In Appendix [A.3](#A1.SS3 "A.3 Synthetic Data Generation ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we provide an overview and a detailed description of our methods for generating synthetic semantic duplicates for each benchmark.

### 3.4 Finetuning on Duplicates

We finetune Olmo3 Instruct on duplicates of the following benchmark datasets: MuSR, Zebralogic, and MBPP.
We use either exact duplicates or semantic duplicates generated as in Section [3.3](#S3.SS3 "3.3 Generating Synthetic Semantic Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
To finetune Olmo3 Instruct (a non-reasoning model) on these datapoints we first get a teacher model to generate Chain-of-Thought (CoT) (Wei et al., [2022](#bib.bib45 "Chain-of-thought prompting elicits reasoning in large language models")) reasoning traces.
We use Opus 4.5 as teacher model, and for MuSR we also experiment with using GPT 4.1-mini.

We take the formatted questions and corresponding CoT answers and use LoRA (Hu et al., [2022](#bib.bib34 "LoRA: low-rank adaptation of large language models")) to finetune Olmo3 Instruct.
To get propensities we use a temperature of 0.7 and do 8 parallel generations (for each of the unfinetuned model, the CoT generations of the teacher model, and for the finetuned model).

We split a finetuning dataset of duplicates in two and only finetune on half of it, while evaluating the finetuned model on both seen and unseen duplicates.
To assess whether performance goes up on related benchmarks we use TrueDetective (Del and Fishel, [2023](#bib.bib36 "True detective: a deep abductive reasoning benchmark undoable for gpt-3 and challenging for gpt-4")) as a mirror for MuSR, Arc Challenge (Clark et al., [2018](#bib.bib39 "Think you have solved question answering? try arc, the ai2 reasoning challenge")) as a mirror for ZebraLogic, and HumanEval as a mirror for MBPP.
We also evaluate performance on Arc Easy, BoolQ (Clark et al., [2019](#bib.bib46 "Boolq: exploring the surprising difficulty of natural yes/no questions")), HellaSwag (Zellers et al., [2019](#bib.bib47 "Hellaswag: can a machine really finish your sentence?")), PIQA (Bisk et al., [2020](#bib.bib48 "Piqa: reasoning about physical commonsense in natural language")) and Winogrande (Sakaguchi et al., [2021](#bib.bib49 "Winogrande: an adversarial winograd schema challenge at scale")).

## 4 Results

### 4.1 Exact duplicates in training corpora

Olmo et al. ([2025](#bib.bib33 "Olmo 3")), the paper introducing Olmo3, lists ZebraLogic in their suite of evaluation benchmarks.
Surprisingly however, we find that the Olmo Instruct RL dataset contains exact duplicates of ZebraLogic problems and solutions.
In Figure [1](#S4.F1 "Figure 1 ‣ 4.1 Exact duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we see that for puzzles of 4×44\times 4 or larger, for  70% or more problems, there exists at least one exact duplicate in the Olmo3 Instruct RL dataset (Dolci-Instruct-RL). In total, we find corpus duplicates for at least 49.5% of dataset tasks.

The model card of Olmo3 benchmarks the performance of Olmo3 Instruct after SFT training as 18%, after DPO training as 28.4%, and after RL training as 32.9%. The model is improved in that order.
We do not find exact or semantic duplicates of ZebraLogic in the DPO dataset, so we ascribe the improvement between SFT training and DPO training to general logical reasoning improvement.
However, the increase from 28.4% to 32.9% (possibly on harder problems) is likely due to directly training on ZebraLogic data.

!(/html/2602.12413/assets/x1.png)

Figure 1: On the y-axis we plot the following statistic: for each ZebraLogic benchmark datapoint we check among the top 10 highest cosine similarity training datapoints if any of those samples is an exact duplicate, we then calculate the proportion of benchmark datapoints (of a given grid size) that have at least one exact duplicate. On the x-axis we plot puzzle grid size.

### 4.2 Natural semantic duplicates in training corpora

Relationship of cosine similarity to semantic duplicate status.
In Figure [2](#S4.F2 "Figure 2 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we plot cosine similarity against semantic duplicate status.
To acquire this plot we select the points in the top 0.1% cosine similarity (in matches between benchmark and training dataset), and randomly sample 100 points.
We then use a language model to assess the semantic duplicate status of these 100 datapoints.
For both MBPP and CodeForces we find that even within the top 0.1% highest cosine similarity matches, semantic duplicates are far more common among the highest cosine similarity matches, see Figure [2](#S4.F2 "Figure 2 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").

!(/html/2602.12413/assets/x2.png)

(a) MBPP

!(/html/2602.12413/assets/x3.png)

(b) CodeForces

Figure 2: Relationship between cosine similarity level and semantic duplication.
For each benchmark datapoint we sample 100 matches from the top 0.1% cosine similarity matches in the training data.
On the x-axis we plot the cosine similarity. On the y-axis we plot the percentage of cosine similarity matches at this level that are true semantic duplicates.
The opaque graph shows the confidence interval: this interval widens when there are fewer samples of a given cosine similarity level.
In red we plot semantic duplicates inclusive of exact duplicates, and in blue exclusive.

!(/html/2602.12413/assets/x4.png)

  

Figure 3: Occurence by elo. On the y-axis we plot the following statistic: for each benchmark datapoint we check among the top 100 cosine similarity training datapoints if any of those samples is a semantic duplicate, we then calculate the proportion of all benchmark datapoints that have at least one semantic duplicate. We plot Elo scores on the x-axis.

For MBPP we do not find exact duplicates, which is why the two graphs overlay perfectly, whereas for CodeForces they come apart.
Below we focus on investigating the top 100 cosine similarity matches for each benchmark datapoint.

Proportion of benchmark datapoints that have at least one semantic duplicate.
We find that 100% of MBPP problems have at least one semantic duplicate in the top 100 cosine similarity training data matches.
We also find at least one semantic duplicate per benchmark datapoint when we randomly sample 100 matches out of the top 0.1% cosine similarity matches.
For CodeForces we find that 77.5% of problems have at least one semantic duplicate in the top 100 cosine similarity matches.
Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") considers the likelihood of a test point having a single semantic duplicate based on its elo, and find that problem difficulty (elo score) does not play a big role. We believe the above numbers are a lower bound: if we checked all the data and not just the top 100 cosine similarity matches, we would likely find more semantic duplicates for CodeForces.

!(/html/2602.12413/assets/x5.png)

(a) MBPP

!(/html/2602.12413/assets/x6.png)

(b) CodeForces

Figure 4: Occurence by training dataset.
On the y-axis: for each benchmark datapoint we check among the top 100 cosine similarity training datapoints if any of those samples is a semantic duplicate, we then calculate the proportion of all benchmark datapoints that have at least one semantic duplicate.
On the x-axis we plot the different training datasets.
The lines show the standard deviation.

!(/html/2602.12413/assets/x7.png)

(a) MBPP

!(/html/2602.12413/assets/x8.png)

(b) CodeForces

Figure 5: Occurence by number of cosine similarity matches investigated. We take the number of semantic duplicates evaluated by being at top-n at each of our dataset comparisons.

Semantic duplicate occurence stratified by training dataset.
In Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we investigate the relationship between where in the training scheme a training dataset is used, and the extent of semantic duplicate contamination.
In Figure [4(a)](#S4.F4.sf1 "Figure 4(a) ‣ Figure 4 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we see that all training datasets have semantic duplicates for MBPP.
In Figure [4(b)](#S4.F4.sf2 "Figure 4(b) ‣ Figure 4 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"), for CodeForces we observe that again, semantic duplicates of problems exist in each dataset, particularly in Dolma, Dolci SFT and Dolci DPO.

Semantic duplicates are sparse and investigating more data leads to more matches.
Previous work has found that n-grams typically miss many semantic duplicates (Yang et al., [2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")), which is why like Yang et al. ([2023](#bib.bib27 "Rethinking benchmark and contamination for language models with rephrased samples")) we instead work with cosine similarity matches.
In Figure [6](#S4.F6 "Figure 6 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we see that high cosine similarity matches are sparse.
We also see in Figure [2](#S4.F2 "Figure 2 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") that a substantial portion of the very high (over 0.8) cosine similarity matches is a semantic duplicate.
We also found that for MBPP and CodeForces respectively 100% and 77.5% of problems have at least one semantic duplicate in their top 100 cosine similarity matches.
In Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we see that this statistic - the proportion of benchmark problems that have at least one semantic duplicate in the training data - scales with the number of datapoints we sample.
For example, for CodeForces this statistic drops to 28.4% if we only sample the single top cosine similarity match.
We suggest that the reason we found more semantic duplicates than previous work is that we investigated far more data. Both embedding very large amounts of training data and annotating cosine similarity matches with semantic duplicate status are computationally expensive, so scaling up investigations of this kind is challenging.
See Appendix [B.2](#A2.SS2 "B.2 Semantic duplicates are hard to detect ‣ Appendix B Further Semantic Duplicates in the Wild Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").

!(/html/2602.12413/assets/x9.png)

(a) MBPP, Dolma

!(/html/2602.12413/assets/x10.png)

(b) MBPP, Dolmino

!(/html/2602.12413/assets/x11.png)

(c) MBPP, Dolci SFT

!(/html/2602.12413/assets/x12.png)

(d) MBPP, Dolci DPO

!(/html/2602.12413/assets/x13.png)

(e) MBPP, Dolci RL

!(/html/2602.12413/assets/x14.png)

(f) CodeF, Dolma

!(/html/2602.12413/assets/x15.png)

(g) CodeF, Dolmino

!(/html/2602.12413/assets/x16.png)

(h) CodeF, Dolci SFT

!(/html/2602.12413/assets/x17.png)

(i) CodeF, Dolci DPO

!(/html/2602.12413/assets/x18.png)

(j) CodeF, Dolci RL

Figure 6: Each plot shows cosine similarity distribution of pairs of benchmark data and training corpus data. The top row shows distributions for MBPP and the bottom for CodeForces. From left to right we plot Dolma, Dolmino, Dolci SFT, Dolci DPO and Dolci RL.

### 4.3 Finetuning on Semantic Duplicates

MuSR.
We experimented with three levels of sophistication for semantic duplication.
In Table [4](#S4.T4 "Table 4 ‣ 4.3 Finetuning on Semantic Duplicates ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization")
we find that when we finetune on duplicates of half of the benchmark, performance goes up equally on the unseen half of the benchmark.
We find that finetuning on exact duplicates of MuSR benchmark data leads to a similar increase in performance as finetuning on a variety of types of semantic duplicates.
In both cases the performance goes up by about 20%.
When instead of finetuning on duplicates (exact or semantic), we finetune on datapoints that have been selected for high cosine similarity to the benchmark datapoints, the performance hardly goes up from baseline.
We also check performance change on a same domain but different benchmark, TrueDetective, and find that performance remains stable.
In Appendix [C](#A3 "Appendix C Further Finetuning Results, Including Degradation Analysis ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") we find that when we use a better teacher model to generate CoT reasoning traces, Olmo3 does better after finetuning on them.

ZebraLogic.
In Table [4](#S4.T4 "Table 4 ‣ 4.3 Finetuning on Semantic Duplicates ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
when finetuning on exact duplicates of half of the benchmark data, performance also goes up on the other half.
For ZebraLogic we find that exact duplicates lead to a much larger jump in performance (for both seen and unseen benchmark items) than finetuning on semantic duplicates, in contrast to our finding for MuSR.
In fact, we find that finetuning on semantic duplicates hardly increases performance, and in one case (combining shuffling, substituting and paraphrasing), that the performance on ZebraLogic substantially degrades, even though performance on general datasets remains stable, see Appendix [C](#A3 "Appendix C Further Finetuning Results, Including Degradation Analysis ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
We investigate performance change on a same domain but different benchmark, Arc Challenge, and find that performance is unaffected by finetuning.

MBPP.
In Table [4](#S4.T4 "Table 4 ‣ 4.3 Finetuning on Semantic Duplicates ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"), when finetuning on exact duplicates of half of the benchmark data, we see a substantial jump on that half of the data, while performance on unseen benchmark data hardly changes.
Finetuning on semantic duplicates has a more moderate effect, but affects both seen and unseen performance.
Again, finetuning on cosine similar data does not improve performance substantially.
We also evaluate on a same domain but different coding benchmark, HumanEval, and find a surprising jump for semantic duplicates.
We hypothesize that this jump happened because our semantic duplicate dataset is fairly rich and high quality, and demands some generalization.
We do want to note that our MBPP results are a little noisy: when we stratify the baseline and cosine similarity sft performance evaluation
by the first half of the benchmark data (‘seen’) and second half (‘unseen’)
we find a discrepancy of  6% (in opposite directions: baseline scores higher on second half,
and the cosine similarity model scores higher on the first), when you would expect

Table 2: MuSR

|  |  |  |  |
| --- | --- | --- | --- |
| Duplication  level | Seen | Unseen | True  Detective |
| Baseline |  | 66.0 | 28.3 |
| Exact Dupes | 87.9 | 87.3 | 27.7 |
| Level 1 | 85.8 | 86.2 | 29.3 |
| Level 2 | 85.7 | 86.0 | 28.3 |
| Level 3 | 87.5 | 87.9 | 29.8 |
| Cos Sim sft |  | 68.6 | 25.1 |
| Cos Sim dpo |  | 67.9 | 26.7 |
| Cos Sim rl |  | 65.3 | 26.7 |

Table 3: ZebraLogic

|  |  |  |  |
| --- | --- | --- | --- |
| Duplication  level | Seen | Unseen | Arc  Challenge |
| Baseline |  | 36.9 | 50.1 |
| Exact Dupes | 48.4 | 43.4 | 49.5 |
| Para | 39.2 | 36.2 | 49.3 |
| Shuffle, subs | 36.0 | 36.8 | 50.7 |
| Shuffle, para | 38.0 | 36.0 | 49.4 |
| Shuffle, subs, para | 28.0 | 28.4 | 50.4 |
| Cos Sim sft |  | 22.9 | - |

Table 4: MBPP

|  |  |  |  |
| --- | --- | --- | --- |
| Duplication  level | Seen | Unseen | HumanEval |
| Baseline |  | 46.4 | 55.3 |
| Exact Dupes | 63.0 | 48.8 | 49.2 |
| Semantic Dupes (Py) | 55.1 | 53.6 | 67.0 |
| Cos Sim sft |  | 48.8 | 53.1 |

Table 5: We report on baseline (before finetuning) accuracy on MuSR. We then finetune on 10.000 datapoints.
We either finetune on half of the level 2 & 3 semantic duplicates mixed in with regular data (contaminated model) or we finetune on clean data only (clean model).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Treatment | Seen | Unseen | True  Detective |
| Olmo3 | Baseline |  | 42.8 | 29.3 |
| Finetuned Clean |  | 50.0 | 28.0 |
| Finetuned Contaminated | 66.4 | 54.4 | 28.0 |
| Qwen3 | Baseline |  | 40.4 | 24.0 |
| Finetuned Clean |  | 53.6 | 24.0 |
| Finetuned Contaminated | 65.6 | 52.0 | 28.0 |

performance on these two halves to be similar.

We observe a pattern of shallow generalization.
We repeatedly find that when finetuning on duplicates of benchmark data has a substantial effect on benchmark performance, then performance also improves on benchmark data that was unseen during finetuning.
This suggests within-benchmark-distribution generalization.
We tested benchmark improvement on different, but same domain benchmarks, and typically did not find substantial improvement, confirming shallow generalization.
We also find that improvement or finetuning on high cosine similar datapoints does not by itself improve benchmark performance.

### 4.4 Ecologically Valid Finetuning

Ecologically valid contamination amount.
We evaluate the impact of semantic duplicate contamination under realistic model-developer conditions.
We used an ecologically valid amount of contamination, determined as follows: For MBPP we found that, when we randomly sampled 100 matches among the top 0.1% highest cosine similarity training datapoints for a given benchmark datapoint, on average there were 40 semantic duplicates among the 100 samples.
We concluded that roughly 4 in 10,000 training datapoints are a semantic duplicate for a given benchmark datapoint.

Finetuning contaminated and clean models.
We finetune Olmo3 on data containing MuSR semantic duplicates, for which our annotation experiments verify that no duplicates exist in the model’s training data.
We then split the MuSR data into two halves of 125 datapoints each, and generate 4 semantic duplicates for them, two duplicates of level 2 and two of level 3, so 500 duplicates in total.
We perform two finetuning runs with 1) a clean dataset of 10,000 SFT datapoints verified to be decontaminated, and 2) a contaminated version of the same dataset where 5% of clean samples are swapped with 500 semantic duplicates corresponding to the ‘seen’ subset.
We evaluate both finetuned models on the full MuSR benchmark, splitting results to ‘seen’ and ‘unseen’ MuSR for the contaminated but not for the clean model.

Results on Olmo3.
We find that while the contamination percentage is very low, the contaminated model score on the seen subset is 12% higher than the clean model’s benchmark score, and the contaminated model score on the unseen subset is 5.6% higher that the clean model’s benchmark score.
To verify that performance gains on benchmark items reflects ‘shallow’ generalization rather robust capability gains, we evaluate on TrueDetective, a benchmark in the same domain as MuSR.
We find that performance on TrueDetective remains stable during finetuning.

Noisy results on Qwen3.
Replicating the experiment on Qwen3-8B-base yielded noisy results, possibly due to training instability during finetuning. The contaminated model scored 13.6% higher on seen samples versus unseen, similar to Olmo3. The clean model unexpectedly also showed substantive MuSR benchmark gains from fine-tuning. We note that, puzzlingly, when breaking down the gains of clean finetuning by subdataset we see that the clean model’s gains are particularly strong on the random benchmark-subset we use as the ‘unseen’ subset for the contaminated model, see Appendix [D](#A4 "Appendix D Ecologically Finetuned Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
For Qwen3 we also see some improvement in performance on the same-domain benchmark TrueDetective for the finetuned contaminated model.

Ecologically valid contamination leads to benchmark performance improvement.
Our results show that the presence of semantic duplicates in training corpora, even at low rates, can lead to substantial gains in evaluation results. Breaking the gains down by type, the evidence for gains on ‘seen’ (as semantic duplicates) data from realistic quantities of contamination is strong, while the evidence for within-benchmark generalization from realistic quantities of contamination is mixed. We note that this is the first ecologically valid demonstration of gains on ‘seen’ data from finetuning on semantic duplicates, which previous work only demonstrated in proof-of-concept experiments.
In light of the very strong within-benchmark generalization we observed in the more artificial setting of Section [4.3](#S4.SS3 "4.3 Finetuning on Semantic Duplicates ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"), we believe the topic of within-benchmark generalization in realistic training-setting requires further study.
See Appendix [B.2.1](#A2.SS2.SSS1 "B.2.1 Ecologically valid finetuning experiment ‣ B.2 Semantic duplicates are hard to detect ‣ Appendix B Further Semantic Duplicates in the Wild Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") for more details.

## 5 Limitations and Future Work

We likely underestimate the prevalence of semantic duplicates in real training corpora, because our detection methods in Section [4.2](#S4.SS2 "4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization") are likey to have a high false negative rate.
Another downward bias is the relative absence of rephrasings in Dolma: synthetic data pipelines now often involve intentionally creating semantic duplicates (Wei and Zou, [2019](#bib.bib14 "EDA: easy data augmentation techniques for boosting performance on text classification tasks"); Wang et al., [2022](#bib.bib13 "Self-instruct: aligning language models with self-generated instructions"); Maini et al., [2024](#bib.bib15 "Rephrasing the web: a recipe for compute and data-efficient language modeling")), and so our estimates of their prevalence are likely lower than the true rate in closed corpora using such methods. Similarly, we do not cover state of the art synthetic data provided via RL environments.

Our experiment design is limited to models which open-source their training corpus - a small and potentially unrepresentative set of systems. For instance, the training corpora used in frontier models are much larger than that of the Olmo models – see e.g. the 30T tokens used in the largest Llama 4 runs, (Meta AI, [2025](#bib.bib2 "The Llama 4 herd: the beginning of a new era of natively multimodal AI innovation")). These larger corpora will have more semantic duplicates, but also a different rate of natural semantic duplicates than Dolma.

A fundamental objection to our project could be that out-of-distribution (OOD) generalization is no longer the (only) goal of AI development: an alternative is to instead bring all common tasks in-distribution ([Chollet](#bib.bib8 "The question of whether LLMs can reason is, in many ways, the wrong question [Tweet]") [2024](#bib.bib8 "The question of whether LLMs can reason is, in many ways, the wrong question [Tweet]"), [Patel and Sutton](#bib.bib7 "Richard sutton – father of RL thinks LLMs are a dead end") [2025](#bib.bib7 "Richard sutton – father of RL thinks LLMs are a dead end"), [Leech et al.](#bib.bib9 "Questionable practices in machine learning") [2024](#bib.bib9 "Questionable practices in machine learning") §5.3.2). One could argue that the real-world utility of LLMs shows that our concerns about OOD generalization are not practically important even if generalization is largely shallow. This is a valid perspective, but 1) then the deviation from the assumptions of empirical risk minimization should be explicitly noted, 2) it’s unclear to what extent even perfect hidden interpolation would be practically equivalent to true OOD generalization.

## Impact Statement

We aim here to advance understanding of how LLM benchmark scores relate to general capabilities. Our findings have implications for how the AI research community, policymakers, and the public interpret reported progress on reasoning benchmarks.

More accurately measuring AI capabilities supports calibrated decisions about AI deployment, regulation, and research. If benchmark gains partly reflect interpolation from a growing corpus rather than more general capability improvements, then recognizing this could help prevent overconfidence in model generalization to novel tasks.

We do not believe this work poses significant risks. While our methods could inform more sophisticated benchmark gaming, the contamination we study is likely accidental rather than adversarial, and our detection methods are likely more useful for auditing than for evasion.

Our finding exact and soft contamination in Olmo3 is only possible because of the unusual level of transparency of its model development process. It would be unfair for readers to thereby assume that the level of contamination in Olmo is unusually severe, and worse, a perverse incentive against transparency.

Our work could easily be misread as ‘debunking’ LLM capabilities and so spur complacency about near-term AI impacts. We emphasize that our results suggest that benchmark gains are confounded (and so partially shallow), not that they are illusory.

## Author Contributions

Spiesberger: experimental design for corpus search, embedding, and ecological finetuning experiments; managed compute infrastructure and code repository; executed embedding, MBPP finetuning, and ecological finetuning experiments; designed figures.

Vazquez: experimental design for generation and validation of semantic duplicates; comparison of duplicate detection methods; designed and executed annotation experiments; data spot-checking; wrote parts of methodology, results, and appendices; designed figures.

Pochinkov: generated MuSR teacher examples; finetuned and evaluated Olmo3 models on MuSR and ZebraLogic; data sanity checks.

Gavenčiak: generated synthetic semantic duplicates; contributed to methodology, annotation pipeline design, finetuning experiments, and data analysis; provided feedback on the manuscript.

Grietzer: wrote the introduction and related work sections; edited the manuscript.

Leech: original research question; assembled the team; experiment design; wrote abstract, limitations, future work, and impact statement; secured resources and compute for the project.

Schoots: research and project management; led experiment design; coordinated writing efforts and wrote several parts of the manuscript.

## Acknowledgments

We thank David Latshaw II, Max Shen, and Mary Putt for reading the paper and providing comments. We also thank Owain Evans, John Burden, Tom Davidson, Yuxi Liu, Teortaxes, and Martin Vlach for comments on an earlier draft.
Finally, we thank the Baby, and Jasper for emotional support.

## References

* Anthropic (2025a)
  System card: claude haiku 4.5.
  Note: Model Card
  External Links: [Link](https://www-cdn.anthropic.com/7aad69bf12627d42234e01ee7c36305dc2f6a970.pdf)
  Cited by: [§A.3.1](#A1.SS3.SSS1.p6.1 "A.3.1 Overview ‣ A.3 Synthetic Data Generation ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Anthropic (2025b)
  System card: claude opus 4.5.
  Note: <https://www.anthropic.com/claude-opus-4-5-system-card>Accessed 2026-01-26
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Anthropic (2025c)
  System card: claude sonnet 4.5.
  Note: Model Card
  External Links: [Link](https://www-cdn.anthropic.com/963373e433e489a87a10c823c52a0a013e9172dd.pdf)
  Cited by: [§A.3.1](#A1.SS3.SSS1.p6.1 "A.3.1 Overview ‣ A.3 Synthetic Data Generation ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* J. Austin, A. Odena, M. I. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. J. Cai, M. Terry, Q. V. Le, and C. Sutton (2021)
  Program synthesis with large language models.
  CoRR abs/2108.07732.
  External Links: [Link](https://arxiv.org/abs/2108.07732),
  2108.07732
  Cited by: [§3.1](#S3.SS1.p2.1.1 "3.1 Benchmarks ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Y. Babakhin, R. Osmulski, R. Ak, G. Moreira, M. Xu, B. Schifferer, B. Liu, and E. Oldridge (2025)
  Llama-embed-nemotron-8b: a universal text embedding model for multilingual and cross-lingual tasks.
  arXiv preprint arXiv:2511.07025.
  Cited by: [§3.2](#S3.SS2.p2.1 "3.2 Finding Semantic Duplicates in the Wild ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. (2020)
  Piqa: reasoning about physical commonsense in natural language.
  In Proceedings of the AAAI conference on artificial intelligence,
  Vol. 34,  pp. 7432–7439.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
  Evaluating large language models trained on code.
  External Links: 2107.03374,
  [Link](https://arxiv.org/abs/2107.03374)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* F. Chollet (2024)
  The question of whether LLMs can reason is, in many ways, the wrong question [Tweet].
  Note: X (formerly Twitter)Accessed: 2026-01-26
  External Links: [Link](https://x.com/fchollet/status/1816954290227089656)
  Cited by: [§5](#S5.p3.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova (2019)
  Boolq: exploring the surprising difficulty of natural yes/no questions.
  arXiv preprint arXiv:1905.10044.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  arXiv preprint arXiv:1803.05457.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. (2021)
  Training verifiers to solve math word problems.
  arXiv preprint arXiv:2110.14168.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Del and M. Fishel (2023)
  True detective: a deep abductive reasoning benchmark undoable for gpt-3 and challenging for gpt-4.
  In Proceedings of the 12th Joint Conference on Lexical and Computational Semantics (\* SEM 2023),
   pp. 314–322.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Y. Edelman and J. Lee (2025)
  AI capabilities progress has sped up.
  Note: Accessed: 2026-01-26
  External Links: [Link](https://epoch.ai/data-insights/ai-capabilities-progress-has-sped-up)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Y. Elazar, A. Bhagia, I. H. Magnusson, A. Ravichander, D. Schwenk, A. Suhr, E. P. Walsh, D. Groeneveld, L. Soldaini, S. Singh, H. Hajishirzi, N. A. Smith, and J. Dodge (2024)
  What’s in my big data?.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=RvfPnOkPV4)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Epoch AI (2025)
  Data on ai models.
  Note: Accessed: 2026-01-26
  External Links: [Link](https://epoch.ai/data/ai-models)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy (2020)
  The pile: an 800gb dataset of diverse text for language modeling.
  External Links: 2101.00027,
  [Link](https://arxiv.org/abs/2101.00027)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Google DeepMind (2025)
  Gemini 3 flash model card.
  Note: Model Card
  External Links: [Link](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf)
  Cited by: [§3.2](#S3.SS2.p7.1 "3.2 Finding Semantic Duplicates in the Wild ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2020)
  Measuring massive multitask language understanding.
  arXiv preprint arXiv:2009.03300.
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022)
  LoRA: low-rank adaptation of large language models.
  In The Tenth International Conference on Learning Representations, ICLR
  2022, Virtual Event, April 25-29, 2022,
  External Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9)
  Cited by: [§3.4](#S3.SS4.p2.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Jiang, K. Z. Liu, M. Zhong, R. Schaeffer, S. Ouyang, J. Han, and S. Koyejo (2024)
  Investigating data contamination for pre-training language models.
  External Links: 2401.06059,
  [Document](https://dx.doi.org/10.48550/arXiv.2401.06059),
  [Link](https://arxiv.org/abs/2401.06059)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Y. Kocyigit, E. Briakou, D. Deutsch, J. Luo, C. Cherry, and M. Freitag (2025)
  Overestimation in LLM evaluation: a controlled large-scale study on data contamination’s impact on machine translation.
  In Proceedings of the 42nd International Conference on Machine Learning,
  Proceedings of Machine Learning Research, Vol. 267,  pp. 31105–31132.
  External Links: [Link](https://proceedings.mlr.press/v267/kocyigit25a.html)
  Cited by: [§2](#S2.p5.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* G. Leech, J. J. Vazquez, N. Kupper, M. Yagudin, and L. Aitchison (2024)
  Questionable practices in machine learning.
  External Links: 2407.12220,
  [Link](https://arxiv.org/abs/2407.12220)
  Cited by: [§5](#S5.p3.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* R. Li, L. Ben Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y. Zhuo, T. Wang, O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Monteiro, O. Shliazhko, N. Gontier, N. Meade, A. Zebaze, M. Yee, L. K. Umapathi, J. Zhu, B. Lipkin, M. Oblokulov, Z. Wang, R. Murthy, J. Stillerman, S. S. Patel, D. Abulkhanov, M. Zocca, M. Dey, Z. Zhang, N. Fahmy, U. Bhattacharyya, W. Yu, S. Singh, S. Luccioni, P. Villegas, M. Kunakov, F. Zhdanov, M. Romero, T. Lee, N. Timor, J. Ding, C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao, M. Mishra, A. Gu, J. Robinson, C. J. Anderson, B. Dolan-Gavitt, D. Contractor, S. Reddy, D. Fried, D. Bahdanau, Y. Jernite, C. M. Ferrandis, S. Hughes, T. Wolf, A. Guha, L. von Werra, and H. de Vries (2023)
  StarCoder: may the source be with you!.
  External Links: 2305.06161,
  [Document](https://dx.doi.org/10.48550/arXiv.2305.06161),
  [Link](https://arxiv.org/abs/2305.06161)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* B. Y. Lin, R. L. Bras, K. Richardson, A. Sabharwal, R. Poovendran, P. Clark, and Y. Choi (2025)
  ZebraLogic: on the scaling limits of llms for logical reasoning.
  In Forty-second International Conference on Machine Learning, ICML
  2025, Vancouver, BC, Canada, July 13-19, 2025,
  External Links: [Link](https://openreview.net/forum?id=sTAJ9QyA6l)
  Cited by: [§3.1](#S3.SS1.p5.1.1 "3.1 Benchmarks ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* I. Magar and R. Schwartz (2022)
  Data contamination: from memorization to exploitation.
  In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),
  Dublin, Ireland,  pp. 157–165.
  External Links: [Link](https://aclanthology.org/2022.acl-short.18/),
  [Document](https://dx.doi.org/10.18653/v1/2022.acl-short.18)
  Cited by: [Table 1](#S1.T1.2.2.5.1 "In 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* P. Maini, S. Seto, H. Bai, D. Grangier, Y. Zhang, and N. Jaitly (2024)
  Rephrasing the web: a recipe for compute and data-efficient language modeling.
  External Links: 2401.16380,
  [Link](https://arxiv.org/abs/2401.16380)
  Cited by: [§5](#S5.p1.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* N. Maslej, L. Fattorini, R. Perrault, Y. Gil, V. Parli, N. Kariuki, E. Capstick, A. Reuel, E. Brynjolfsson, J. Etchemendy, K. Ligett, T. Lyons, J. Manyika, J. C. Niebles, Y. Shoham, R. Wald, T. Walsh, A. Hamrah, L. Santarlasci, J. B. Lotufo, A. Rome, A. Shi, and S. Oak (2025)
  Artificial intelligence index report 2025.
  External Links: 2504.07139,
  [Link](https://arxiv.org/abs/2504.07139)
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Meta AI (2025)
  The Llama 4 herd: the beginning of a new era of natively multimodal AI innovation.
  Note: <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>Accessed: 2026-01-26
  Cited by: [§5](#S5.p2.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* N. Muennighoff, N. Tazi, L. Magne, and N. Reimers (2023)
  Mteb: massive text embedding benchmark.
  In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics,
   pp. 2014–2037.
  Cited by: [§3.2](#S3.SS2.p2.1 "3.2 Finding Semantic Duplicates in the Wild ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* T. Olmo, :, A. Ettinger, A. Bertsch, B. Kuehl, D. Graham, D. Heineman, D. Groeneveld, F. Brahman, F. Timbers, H. Ivison, J. Morrison, J. Poznanski, K. Lo, L. Soldaini, M. Jordan, M. Chen, M. Noukhovitch, N. Lambert, P. Walsh, P. Dasigi, R. Berry, S. Malik, S. Shah, S. Geng, S. Arora, S. Gupta, T. Anderson, T. Xiao, T. Murray, T. Romero, V. Graf, A. Asai, A. Bhagia, A. Wettig, A. Liu, A. Rangapur, C. Anastasiades, C. Huang, D. Schwenk, H. Trivedi, I. Magnusson, J. Lochner, J. Liu, L. J. V. Miranda, M. Sap, M. Morgan, M. Schmitz, M. Guerquin, M. Wilson, R. Huff, R. L. Bras, R. Xin, R. Shao, S. Skjonsberg, S. Z. Shen, S. S. Li, T. Wilde, V. Pyatkin, W. Merrill, Y. Chang, Y. Gu, Z. Zeng, A. Sabharwal, L. Zettlemoyer, P. W. Koh, A. Farhadi, N. A. Smith, and H. Hajishirzi (2025)
  Olmo 3.
  External Links: 2512.13961,
  [Link](https://arxiv.org/abs/2512.13961)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§3](#S3.p1.1 "3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§4.1](#S4.SS1.p1.1 "4.1 Exact duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Ł. Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, J. H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Ł. Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mély, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. H. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. B. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph (2024)
  GPT-4 technical report.
  External Links: 2303.08774,
  [Link](https://arxiv.org/abs/2303.08774)
  Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* D. Patel and R. Sutton (2025)
  Richard sutton – father of RL thinks LLMs are a dead end.
  Note: Dwarkesh PodcastPodcast interview
  External Links: [Link](https://www.dwarkesh.com/p/richard-sutton)
  Cited by: [§5](#S5.p3.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* G. Penedo, A. Lozhkov, H. Kydlíček, L. B. Allal, E. Beeching, A. P. Lajarín, Q. Gallouédec, N. Habib, L. Tunstall, and L. von Werra (2025)
  CodeForces.
   Hugging Face.
  Note: <https://huggingface.co/datasets/open-r1/codeforces>
  Cited by: [§3.1](#S3.SS1.p3.1.1 "3.1 Benchmarks ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Riddell, A. Ni, and A. Cohan (2024)
  Quantifying contamination in evaluating code generation capabilities of language models.
  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.),
  Bangkok, Thailand,  pp. 14116–14137.
  External Links: [Link](https://aclanthology.org/2024.acl-long.761/),
  [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.761)
  Cited by: [Table 1](#S1.T1.1.1.1.2 "In 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§1](#S1.p3.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§1](#S1.p4.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p5.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi (2021)
  Winogrande: an adversarial winograd schema challenge at scale.
  Communications of the ACM 64 (9),  pp. 99–106.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, and L. Zettlemoyer (2023)
  Detecting pretraining data from large language models.
  External Links: 2310.16789,
  [Link](https://arxiv.org/abs/2310.16789)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* I. Shilov, M. Meeus, and Y. de Montjoye (2025)
  The mosaic memory of large language models.
  Note: arXiv:2405.15523v2
  External Links: 2405.15523,
  [Document](https://dx.doi.org/10.48550/arXiv.2405.15523),
  [Link](https://arxiv.org/abs/2405.15523)
  Cited by: [Table 1](#S1.T1.2.2.2.2 "In 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* L. Soldaini, R. Kinney, A. Bhagia, D. Schwenk, D. Atkinson, R. Authur, B. Bogin, K. Chandu, J. Dumas, Y. Elazar, V. Hofmann, A. Jha, S. Kumar, L. Lucy, X. Lyu, N. Lambert, I. Magnusson, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. Peters, A. Ravichander, K. Richardson, Z. Shen, E. Strubell, N. Subramani, O. Tafjord, E. Walsh, L. Zettlemoyer, N. Smith, H. Hajishirzi, I. Beltagy, D. Groeneveld, J. Dodge, and K. Lo (2024)
  Dolma: an open corpus of three trillion tokens for language model pretraining research.
  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.),
  Bangkok, Thailand,  pp. 15725–15788.
  External Links: [Link](https://aclanthology.org/2024.acl-long.840/),
  [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.840)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Z. Sprague, X. Ye, K. Bostrom, S. Chaudhuri, and G. Durrett (2024)
  MuSR: testing the limits of chain-of-thought with multistep soft reasoning.
  In The Twelfth International Conference on Learning Representations,
  ICLR 2024, Vienna, Austria, May 7-11, 2024,
  External Links: [Link](https://openreview.net/forum?id=jenyYQzue1)
  Cited by: [§3.1](#S3.SS1.p4.1.1 "3.1 Benchmarks ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi (2022)
  Self-instruct: aligning language models with self-generated instructions.
  External Links: 2212.10560,
  [Link](https://arxiv.org/abs/2212.10560)
  Cited by: [§5](#S5.p1.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* M. Weber, D. Fu, Q. Anthony, Y. Oren, S. Adams, A. Alexandrov, X. Lyu, H. Nguyen, X. Yao, V. Adams, B. Athiwaratkun, R. Chalamala, K. Chen, M. Ryabinin, T. Dao, P. Liang, C. Ré, I. Rish, and C. Zhang (2024)
  RedPajama: an open dataset for training large language models.
  External Links: 2411.12372,
  [Document](https://dx.doi.org/10.48550/arXiv.2411.12372),
  [Link](https://arxiv.org/abs/2411.12372)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)
  Chain-of-thought prompting elicits reasoning in large language models.
  Advances in neural information processing systems 35,  pp. 24824–24837.
  Cited by: [§3.4](#S3.SS4.p1.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* J. Wei and K. Zou (2019)
  EDA: easy data augmentation techniques for boosting performance on text classification tasks.
  In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan (Eds.),
  Hong Kong, China,  pp. 6382–6388.
  External Links: [Link](https://aclanthology.org/D19-1670/),
  [Document](https://dx.doi.org/10.18653/v1/D19-1670)
  Cited by: [§5](#S5.p1.1 "5 Limitations and Future Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* C. Xu, N. Yan, S. Guan, Y. Mei, and T. Kechadi (2025)
  SSA: semantic contamination of LLM-driven fake news detection.
  In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng (Eds.),
  Suzhou, China,  pp. 14737–14751.
  External Links: [Link](https://aclanthology.org/2025.emnlp-main.744/),
  [Document](https://dx.doi.org/10.18653/v1/2025.emnlp-main.744)
  Cited by: [Table 1](#S1.T1.2.2.7.1 "In 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p5.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* S. Yang, W. Chiang, L. Zheng, J. E. Gonzalez, and I. Stoica (2023)
  Rethinking benchmark and contamination for language models with rephrased samples.
  External Links: 2311.04850,
  [Document](https://dx.doi.org/10.48550/arXiv.2311.04850),
  [Link](https://arxiv.org/abs/2311.04850)
  Cited by: [Table 1](#S1.T1.2.2.6.1 "In 1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§1](#S1.p3.1 "1 Introduction ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p4.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p5.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§4.2](#S4.SS2.p5.1 "4.2 Natural semantic duplicates in training corpora ‣ 4 Results ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)
  Hellaswag: can a machine really finish your sentence?.
  arXiv preprint arXiv:1905.07830.
  Cited by: [§3.4](#S3.SS4.p3.1 "3.4 Finetuning on Duplicates ‣ 3 Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").
* X. Zhou, M. Weyssow, R. Widyasari, T. Zhang, J. He, Y. Lyu, J. Chang, B. Zhang, D. Huang, and D. Lo (2025)
  LessLeak-bench: a first investigation of data leakage in LLMs across 83 software engineering benchmarks.
  External Links: 2502.06215,
  [Document](https://dx.doi.org/10.48550/arXiv.2502.06215),
  [Link](https://arxiv.org/abs/2502.06215)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization"),
  [§2](#S2.p3.1 "2 Related Work ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").

## Appendix A Further Details on Methodology

### A.1 Olmo3 Instruct Training Datasets

We embedded 1% of Dolma3\_6T-mix-1025, the data Olmo3 Base was trained on; and 1% of Dolmino-mix-1025 (100B tokens, 2.7 GB), a high quality dataset Olmo3 Base is trained on, here we excluded long-context data.
We also embedded the following three high quality datasets used to finetune Olmo3 Base into Olmo3 Instruct: Dolci3 SFT (2M input-output prompts, 3.08 GB), Dolci3 Instruct DPO (260k preference pairs, 811 MB), and Dolci3 Instruct RL (169k prompts, 483 MB).

Table 6: Datasets processed for contamination analysis. All text chunks are filtered to 50–2,048 tokens. Pretraining data is sampled at 1% using stratified reservoir sampling to preserve source distribution topology.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Orig. Size | Sample | Sampling Method |
| Pretraining Data | | | |
| Dolma3\_6T-mix | 6.0 TB | 1.0% | Stratified Reservoir (Hierarchical) |
| Dolmino-mix | 250 GB | 1.0% | Stratified Reservoir (Hierarchical) |
| Instruction Tuning Data | | | |
| Dolci3 SFT | 3.08 GB | 100.0% | Full Ingestion |
| Dolci3 DPO | 811 MB | 100.0% | Full Ingestion |
| Dolci3 RL | 483 MB | 100.0% | Full Ingestion |

### A.2 Extended Cosine Similarity discussion

#### A.2.1 MuSR

!(/html/2602.12413/assets/x19.png)

(a) MuSR, Dolma

!(/html/2602.12413/assets/x20.png)

(b) MuSR, Dolmino

!(/html/2602.12413/assets/x21.png)

(c) MuSR, DolciSFT

!(/html/2602.12413/assets/x22.png)

(d) MuSR, DolciDPO

!(/html/2602.12413/assets/x23.png)

(e) MuSR, DolciRL

Figure 7: Each plot shows cosine similarity distribution of pairs of the MuSR benchmark data and training corpus data. From left to right we plot Dolma, Dolmino, Dolci SFT, Dolci DPO and Dolci RL.

#### A.2.2 ZebraLogic

!(/html/2602.12413/assets/x24.png)

(a) ZebraLogic, Dolma

!(/html/2602.12413/assets/x25.png)

(b) ZebraLogic, Dolmino

!(/html/2602.12413/assets/x26.png)

(c) ZebraLogic, DolciSFT

!(/html/2602.12413/assets/x27.png)

(d) ZebraLogic,DolciDPO

!(/html/2602.12413/assets/x28.png)

(e) ZebraLogic, DolciRL

Figure 8: Each plot shows cosine similarity distribution of pairs of ZebraLogic benchmark data and training corpus data. From left to right we plot Dolma, Dolmino, Dolci SFT, Dolci DPO and Dolci RL.

### A.3 Synthetic Data Generation

#### A.3.1 Overview

MBPP

We generate 5 semantic duplicates for both inputs (questions), and outputs (responses).
For the text inputs, we generate paraphrasings simultaneously to maximise difference between the texts.
For code outputs, we generate alternative python implementations and validate them against provided test cases.

MuSR We adapt the original MuSR sample generation logic to generate new samples from existing reasoning trees of the public dataset problems. For the Murder Mysteries and Team Allocation tasks, we create three levels of semantic duplicates differing by how much the underlying logic trees differ from the original while maintaining the answer and the necessary reasoning to solve the problem the same: Level 1) tree is kept the same and story context is regenerated; Level 2) One branch that does not affect problem outcome or its complexity is changed; Level 3) All branches of the previous type are changed.

For each level, we generate 2 semantic duplicates for each of the 250 original samples, adding up to 1500 semantic duplicates per task.
We also generate a new test set of 250 samples using the original code.

ZebraLogic To generate semantic duplicates of ZebraLogic datapoints we use a variety of transformations, most of which are LLM based.

We apply the following transformation methods:
1) use category mappings that e.g. substitute “color” by “shape” and “red” by “square”, using Claude 4.5 Haiku (Anthropic, [2025a](#bib.bib43 "System card: claude haiku 4.5"));
2) shuffle conditions or clues in the prompt using a Python function; and
3) paraphrasing text while preserving all values exactly, using Claude 4.5 Sonnet (Anthropic, [2025c](#bib.bib44 "System card: claude sonnet 4.5")).
These strategies are applied to the first 500 samples in the benchmark, generating semantic duplicates for paraphrasing alone, and the following combinations of the above: 1) and 2); 2) and 3); and 1), 2) and 3).

#### A.3.2 Details of Finetuning on Duplicates

Table 7: Finetuning hyperparameters for each benchmark. All experiments use LoRA Rank 16, Alpha 32, and Dropout 0.05.

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | MBPP | MuSR | ZebraLogic |
| Learning rate | 1.5​e−41.5e{-}4 | 2​e−42e{-}4 | 2​e−42e{-}4 |
| KL penalty | 0.020.02 | – | – |
| Epochs | 10 | 6 | 10 |

#### A.3.3 MuSR

The generation pipeline uses the same few-shot examples and prompt templates as the original benchmark generation set up. The following model parameters are used for with gpt-4-0613:

* •

  Temperature: 1.0
* •

  Top P: 1.0
* •

  Max tokens: 2400

Specific details for each MuSR task:

Murder Mysteries In this task, the model needs to figure out which of the two suspects is the murderer based on a long narrative generated from a logic tree. We make the following modifications to the logic trees before the story generation step:

* •

  Level 0: Story context regenerated from unchanged original tree.
* •

  Level 1: One suspicious fact branches changed.
* •

  Level 2: All branches (suspicious, means, motive, opportunity) belonging to the innocent suspect changed.

Team Allocation In this task, the model needs to determine how to best allocate the 3 individuals mentioned in the long narrative to perform two tasks.
We make the following modifications to the logic trees before the story generation step:

* •

  Level 0: Story context regenerated from unchanged original tree.
* •

  Level 1: One randomly selected skill branch is changed while keeping skill level unchanged.
* •

  Level 2: All branches swapped while skill and cooperation levels unchanged

Object Placement The data available for the original samples is not enough to generate semantic duplicates. While it is possible to attempt to extract most of the required data with LLMs to attempt sample regeneration, format and style is lost, affecting subsequent generation steps in the multi-stage process. So we choose to omit this MuSR category due to difficulty in creating semantic duplicates of similar complexity and rigor.

#### A.3.4 MBPP

We used claude-opus-4-5-20251101 for all generation tasks with parameters:

* •

  max tokens: 10241024
* •

  temperature: 1.0

We generate semantic duplicates for the full MBPP sanitized dataset, that is, 427 tasks. In the process, we find that the following tasks contain bugs (either in the function, or in the test cases): 229229, 438438, 461461, 579579, 769769, 802802.

Inputs We batch generate paraphrasings to ensure clear differences between duplicates.

Outputs We generate Python semantic duplicates for each sample sequentially, allowing the model to see the original and previously generated ones to increase the uniqueness of new solutions. We validate solutions at each step and ensure <0.85<0.85 difflib.SequenceMatcher similarity score between solutions.

Prompts

Prompt for paraphrased question text duplicates:

[⬇](data:text/plain;base64,UEFSQVBIUkFTRV9CQVRDSF9QUk9NUFQgPSAiIiJZb3UgYXJlIGFuIGV4cGVydCBhdCBwYXJhcGhyYXNpbmcgcHJvZ3JhbW1pbmcgdGFzayBkZXNjcmlwdGlvbnMuCgpPUklHSU5BTCBUQVNLOgp7dGV4dH0KCllPVVIgVEFTSzoKR2VuZXJhdGUgZXhhY3RseSA1IERJU1RJTkNUIHBhcmFwaHJhc2VzIG9mIHRoaXMgcHJvZ3JhbW1pbmcgdGFzay4gRWFjaCBwYXJhcGhyYXNlIG11c3Q6CjEuIEhhdmUgQ09NUExFVEVMWSBESUZGRVJFTlQgd29yZGluZyBmcm9tIHRoZSBvdGhlcnMKMi4gUHJlc2VydmUgdGhlIEVYQUNUIHNhbWUgbWVhbmluZyBhbmQgcmVxdWlyZW1lbnRzCjMuIE1haW50YWluIHRoZSBzYW1lIGxldmVsIG9mIHRlY2huaWNhbCBkZXRhaWwgYW5kIGNsYXJpdHkKNC4gS2VlcCBhbnkgbWVudGlvbmVkIGZ1bmN0aW9uIG5hbWVzIFVOQ0hBTkdFRAoKQ1JJVElDQUw6IEVhY2ggcGFyYXBocmFzZSBtdXN0IGJlIG5vdGljZWFibHkgZGlmZmVyZW50IGZyb20gdGhlIG90aGVycy4gVmFyeToKLSBTZW50ZW5jZSBzdHJ1Y3R1cmUgKGFjdGl2ZSB2cyBwYXNzaXZlLCBxdWVzdGlvbnMgdnMgc3RhdGVtZW50cykKLSBWb2NhYnVsYXJ5IGNob2ljZXMgKHN5bm9ueW1zLCBkaWZmZXJlbnQgdGVjaG5pY2FsIHRlcm1zKQotIE9yZGVyIG9mIGluZm9ybWF0aW9uIHByZXNlbnRlZAotIExldmVsIG9mIGZvcm1hbGl0eQoKQVZPSUQ6Ci0gU3RhcnRpbmcgbXVsdGlwbGUgcGFyYXBocmFzZXMgdGhlIHNhbWUgd2F5IChlLmcuLCBkb24ndCBzdGFydCAzIHdpdGggIldyaXRlIGEuLi4iKQotIFNpbXBseSBzd2FwcGluZyBvbmUgb3IgdHdvIHdvcmRzIHdoaWxlIGtlZXBpbmcgc3RydWN0dXJlIGlkZW50aWNhbAotIEFkZGluZyBvciByZW1vdmluZyByZXF1aXJlbWVudHMgbm90IGluIHRoZSBvcmlnaW5hbAotIENoYW5naW5nIHRoZSBwcm9ncmFtbWluZyBsYW5ndWFnZSBpZiBvbmUgaXMgc3BlY2lmaWVkCi0gTWFraW5nIHRoZSB0YXNrIGFtYmlndW91cyBvciBsZXNzIHByZWNpc2UKCk91dHB1dCBFWEFDVExZIGluIHRoaXMgSlNPTiBmb3JtYXQgKG5vIGV4dHJhIHRleHQsIG5vIG1hcmtkb3duKToKe3sKICAicGFyYTEiOiAiZmlyc3QgcGFyYXBocmFzZSBoZXJlIiwKICAicGFyYTIiOiAic2Vjb25kIHBhcmFwaHJhc2UgaGVyZSIsCiAgInBhcmEzIjogInRoaXJkIHBhcmFwaHJhc2UgaGVyZSIsCiAgInBhcmE0IjogImZvdXJ0aCBwYXJhcGhyYXNlIGhlcmUiLAogICJwYXJhNSI6ICJmaWZ0aCBwYXJhcGhyYXNlIGhlcmUiCn19CgpHZW5lcmF0ZSB0aGUgNSBkaXZlcnNlIHBhcmFwaHJhc2VzIG5vdzoiIiI=)

PARAPHRASE\_BATCH\_PROMPT = """You are an expert at paraphrasing programming task descriptions.

ORIGINAL TASK:

{text}

YOUR TASK:

Generate exactly 5 DISTINCT paraphrases of this programming task. Each paraphrase must:

1. Have COMPLETELY DIFFERENT wording from the others

2. Preserve the EXACT same meaning and requirements

3. Maintain the same level of technical detail and clarity

4. Keep any mentioned function names UNCHANGED

CRITICAL: Each paraphrase must be noticeably different from the others. Vary:

- Sentence structure (active vs passive, questions vs statements)

- Vocabulary choices (synonyms, different technical terms)

- Order of information presented

- Level of formality

AVOID:

- Starting multiple paraphrases the same way (e.g., don’t start 3 with "Write a...")

- Simply swapping one or two words while keeping structure identical

- Adding or removing requirements not in the original

- Changing the programming language if one is specified

- Making the task ambiguous or less precise

Output EXACTLY in this JSON format (no extra text, no markdown):

{{

"para1": "first paraphrase here",

"para2": "second paraphrase here",

"para3": "third paraphrase here",

"para4": "fourth paraphrase here",

"para5": "fifth paraphrase here"

}}

Generate the 5 diverse paraphrases now:"""

Prompts for alternative Python solution implementations:

[⬇](data:text/plain;base64,IyBQeXRob24gc2VtYW50aWMgZHVwbGljYXRlIGdlbmVyYXRpb24KZGVmIGdldF9nZW5lcmF0aW9uX3Byb21wdCgKICAgIHRhc2tfZGVzY3JpcHRpb246IHN0ciwKICAgIG9yaWdpbmFsX2NvZGU6IHN0ciwKICAgIHRlc3RfbGlzdDogbGlzdFtzdHJdLAogICAgcHJldmlvdXNfc29sdXRpb25zOiBsaXN0W3N0cl0sCiAgICBwcmV2aW91c19lcnJvcjogT3B0aW9uYWxbc3RyXSA9IE5vbmUsCiAgICByZXF1aXJlX21vcmVfZGlmZmVyZW50OiBib29sID0gRmFsc2UKKSAtPiBzdHI6CiAgICAiIiJHZW5lcmF0ZSB0aGUgcHJvbXB0IGZvciBjcmVhdGluZyBhIFB5dGhvbiBzZW1hbnRpYyBkdXBsaWNhdGUuIiIiCgogICAgdGVzdF9jb250ZXh0ID0gIlxuIi5qb2luKHRlc3RfbGlzdFs6M10pCiAgICBmdW5jX25hbWUgPSBleHRyYWN0X2Z1bmN0aW9uX25hbWUob3JpZ2luYWxfY29kZSkKCiAgICBiYXNlX3Byb21wdCA9IGYiIiJZb3UgYXJlIGFuIGV4cGVydCBQeXRob24gcHJvZ3JhbW1lci4gWW91ciB0YXNrIGlzIHRvIHdyaXRlIGEgRElGRkVSRU5UIFB5dGhvbiBzb2x1dGlvbiBmb3IgdGhlIGZvbGxvd2luZyBwcm9ibGVtLgoKVEFTSyBERVNDUklQVElPTjoKe3Rhc2tfZGVzY3JpcHRpb259CgpPUklHSU5BTCBQWVRIT04gU09MVVRJT04gKGZvciByZWZlcmVuY2UgLSBETyBOT1QgQ09QWSk6CmBgYHB5dGhvbgp7b3JpZ2luYWxfY29kZX0KYGBgCgpQWVRIT04gVEVTVCBFWEFNUExFUyAoeW91ciBzb2x1dGlvbiBtdXN0IHBhc3MgdGhlc2UpOgpgYGBweXRob24Ke3Rlc3RfY29udGV4dH0KYGBgCgoiIiIKCiAgICAjIEFkZCBwcmV2aW91cyBzb2x1dGlvbnMgaWYgYW55CiAgICBpZiBwcmV2aW91c19zb2x1dGlvbnM6CiAgICAgICAgYmFzZV9wcm9tcHQgKz0gIlBSRVZJT1VTIFNPTFVUSU9OUyBZT1UnVkUgQUxSRUFEWSBXUklUVEVOICh5b3VyIG5ldyBzb2x1dGlvbiBtdXN0IGJlIFNUUlVDVFVSQUxMWSBESUZGRVJFTlQgZnJvbSBBTEwgb2YgdGhlc2UpOlxuIgogICAgICAgIGZvciBpLCBzb2wgaW4gZW51bWVyYXRlKHByZXZpb3VzX3NvbHV0aW9ucywgMSk6CiAgICAgICAgICAgIGJhc2VfcHJvbXB0ICs9IGYiXG4tLS0gU29sdXRpb24ge2l9IC0tLVxuYGBgcHl0aG9uXG57c29sfVxuYGBgXG4iCiAgICAgICAgYmFzZV9wcm9tcHQgKz0gIlxuIgoKICAgICMgQWRkIGVycm9yIGZlZWRiYWNrIGlmIHJldHJ5CiAgICBpZiBwcmV2aW91c19lcnJvcjoKICAgICAgICBiYXNlX3Byb21wdCArPSBmIiIiWU9VUiBQUkVWSU9VUyBBVFRFTVBUIEhBRCBFUlJPUlM6CntwcmV2aW91c19lcnJvcn0KClBsZWFzZSBmaXggdGhlc2UgZXJyb3JzIGluIHlvdXIgbmV3IHNvbHV0aW9uLgoKIiIiCgogICAgIyBBZGQgc3Ryb25nZXIgZGlmZmVyZW50aWF0aW9uIHJlcXVlc3QgaWYgbmVlZGVkCiAgICBpZiByZXF1aXJlX21vcmVfZGlmZmVyZW50OgogICAgICAgIGJhc2VfcHJvbXB0ICs9ICIiIklNUE9SVEFOVDogWW91ciBwcmV2aW91cyBzb2x1dGlvbiB3YXMgVE9PIFNJTUlMQVIgdG8gZXhpc3Rpbmcgb25lcyEKWW91IE1VU1QgdXNlIGEgc2lnbmlmaWNhbnRseSBESUZGRVJFTlQgYWxnb3JpdGhtaWMgYXBwcm9hY2guIENvbnNpZGVyOgotIFVzaW5nIGRpZmZlcmVudCBkYXRhIHN0cnVjdHVyZXMgKGxpc3QgdnMgc2V0IHZzIGRpY3QgdnMgZGVxdWUpCi0gVXNpbmcgZGlmZmVyZW50IGl0ZXJhdGlvbiBwYXR0ZXJucyAoZm9yIHZzIHdoaWxlIHZzIHJlY3Vyc2lvbiB2cyBjb21wcmVoZW5zaW9ucykKLSBVc2luZyBkaWZmZXJlbnQgYnVpbHQtaW4gZnVuY3Rpb25zIG9yIGxpYnJhcmllcwotIFJlc3RydWN0dXJpbmcgdGhlIGxvZ2ljIGZsb3cgY29tcGxldGVseQoKIiIiCgogICAgYmFzZV9wcm9tcHQgKz0gZiIiIlJFUVVJUkVNRU5UUzoKMS4gV3JpdGUgYSBDT01QTEVURSBQeXRob24gc29sdXRpb24gdGhhdCBwYXNzZXMgYWxsIHRlc3RzCjIuIFRoZSBmdW5jdGlvbiBNVVNUIGJlIG5hbWVkIEVYQUNUTFk6IHtmdW5jX25hbWV9CjMuIFVzZSBhIERJRkZFUkVOVCBhbGdvcml0aG1pYyBhcHByb2FjaCBvciBpbXBsZW1lbnRhdGlvbiBzdHlsZSB0aGFuIHRoZSBzb2x1dGlvbnMgc2hvd24gYWJvdmUKNC4gQUREIEEgQ09NTUVOVCBPTiBUSEUgTElORSBBQk9WRSBFQUNIIExJTkUgT0YgQ09ERSBleHBsYWluaW5nIHdoYXQgaXQgZG9lcwo1LiBDb21tZW50cyBzaG91bGQgYmUgT1JJR0lOQUwsIENPTkNJU0UsIGFuZCBJTlNJR0hURlVMIC0gbm90IGdlbmVyaWMKNi4gTWFrZSBzdXJlIGV2ZXJ5IHN1YnN0YW50aXZlIGxpbmUgaGFzIGEgY29tbWVudCBhYm92ZSBpdAo3LiBUaGUgY29tbWVudHMgc2hvdWxkIGhlbHAgZGlzdGluZ3Vpc2ggdGhpcyBzb2x1dGlvbiBzZW1hbnRpY2FsbHkgZnJvbSBvdGhlcnMKCkNPTU1FTlQgU1RZTEUgRVhBTVBMRToKYGBgcHl0aG9uCiMgSW5pdGlhbGl6ZSBjb3VudGVyIGZvciB0cmFja2luZyBlbGVtZW50IGZyZXF1ZW5jeQpjb3VudCA9IDAKIyBJdGVyYXRlIHRocm91Z2ggZWFjaCBpdGVtIGluIHRoZSBpbnB1dCBzZXF1ZW5jZQpmb3IgaXRlbSBpbiBpdGVtczoKICAgICMgSW5jcmVtZW50IGNvdW50ZXIgd2hlbiBjb25kaXRpb24gaXMgbWV0CiAgICBpZiBjb25kaXRpb246CiAgICAgICAgY291bnQgKz0gMQojIFJldHVybiB0aGUgZmluYWwgYWNjdW11bGF0ZWQgY291bnQKcmV0dXJuIGNvdW50CmBgYAoKT1VUUFVUIE9OTFkgVEhFIFBZVEhPTiBDT0RFIHdpdGggY29tbWVudHMuIE5vIG1hcmtkb3duIGNvZGUgYmxvY2tzLCBubyBleHBsYW5hdGlvbnMgb3V0c2lkZSB0aGUgY29kZS4iIiIKCiAgICByZXR1cm4gYmFzZV9wcm9tcHQ=)

# Python semantic duplicate generation

def get\_generation\_prompt(

task\_description: str,

original\_code: str,

test\_list: list[str],

previous\_solutions: list[str],

previous\_error: Optional[str] = None,

require\_more\_different: bool = False

) -> str:

"""Generate the prompt for creating a Python semantic duplicate."""

test\_context = "\n".join(test\_list[:3])

func\_name = extract\_function\_name(original\_code)

base\_prompt = f"""You are an expert Python programmer. Your task is to write a DIFFERENT Python solution for the following problem.

TASK DESCRIPTION:

{task\_description}

ORIGINAL PYTHON SOLUTION (for reference - DO NOT COPY):

‘‘‘python

{original\_code}

‘‘‘

PYTHON TEST EXAMPLES (your solution must pass these):

‘‘‘python

{test\_context}

‘‘‘

"""

# Add previous solutions if any

if previous\_solutions:

base\_prompt += "PREVIOUS SOLUTIONS YOU’VE ALREADY WRITTEN (your new solution must be STRUCTURALLY DIFFERENT from ALL of these):\n"

for i, sol in enumerate(previous\_solutions, 1):

base\_prompt += f"\n--- Solution {i} ---\n‘‘‘python\n{sol}\n‘‘‘\n"

base\_prompt += "\n"

# Add error feedback if retry

if previous\_error:

base\_prompt += f"""YOUR PREVIOUS ATTEMPT HAD ERRORS:

{previous\_error}

Please fix these errors in your new solution.

"""

# Add stronger differentiation request if needed

if require\_more\_different:

base\_prompt += """IMPORTANT: Your previous solution was TOO SIMILAR to existing ones!

You MUST use a significantly DIFFERENT algorithmic approach. Consider:

- Using different data structures (list vs set vs dict vs deque)

- Using different iteration patterns (for vs while vs recursion vs comprehensions)

- Using different built-in functions or libraries

- Restructuring the logic flow completely

"""

base\_prompt += f"""REQUIREMENTS:

1. Write a COMPLETE Python solution that passes all tests

2. The function MUST be named EXACTLY: {func\_name}

3. Use a DIFFERENT algorithmic approach or implementation style than the solutions shown above

4. ADD A COMMENT ON THE LINE ABOVE EACH LINE OF CODE explaining what it does

5. Comments should be ORIGINAL, CONCISE, and INSIGHTFUL - not generic

6. Make sure every substantive line has a comment above it

7. The comments should help distinguish this solution semantically from others

COMMENT STYLE EXAMPLE:

‘‘‘python

# Initialize counter for tracking element frequency

count = 0

# Iterate through each item in the input sequence

for item in items:

# Increment counter when condition is met

if condition:

count += 1

# Return the final accumulated count

return count

‘‘‘

OUTPUT ONLY THE PYTHON CODE with comments. No markdown code blocks, no explanations outside the code."""

return base\_prompt

#### A.3.5 ZebraLogic

Below we discuss the different methods and transformations:

Paraphrasing. With claude-4.5-sonnet we use the following prompts using the original sample

[⬇](data:text/plain;base64,IyBTWVNURU0gUFJPTVBUCllvdSBhcmUgYW4gZXhwZXJ0IGVkaXRvciB0YXNrZWQgd2l0aCByZXdyaXRpbmcgbG9naWMgZ3JpZCBwdXp6bGVzIHdoaWxlIGV4YWN0bHkgcHJlc2VydmluZyB0aGUgbG9naWNhbCBzdHJ1Y3R1cmUgYW5kIHNlbWFudGljcy4KCiMgVVNFUiBQUk9NUFQKUmV3cml0ZSB0aGUgZm9sbG93aW5nIGxvZ2ljIHB1enpsZSB0byBleHByZXNzIHRoZSBleGFjdCBzYW1lIGNvbmRpdGlvbnMgaW4gZGlmZmVyZW50IHdvcmRzIG9yIHdpdGggZGlmZmVyZW50IHdvcmQgb3JkZXIgZXRjLiB3aGlsZSBleGFjdGx5IHByZXNlcnZpbmcgdGhlIGxvZ2ljYWwgc3RydWN0dXJlIGFuZCBzZW1hbnRpY3MuCgpPcmlnaW5hbCBQdXp6bGU6CntwdXp6bGV9CgpSRVFVSVJFTUVOVFM6CjEuIFJlZm9ybXVsYXRlIGJvdGggdGhlIHRhc2sgZGVzY3JpcHRpb24gYW5kIGV2ZXJ5IG51bWJlcmVkIGNvbmRpdGlvbi4KMi4gWW91IG1heSBjaGFuZ2Ugd29yZCBvcmRlciwgdXNlIHN5bm9ueW1zLCBhbmQgYWx0ZXIgc2VudGVuY2Ugc3RydWN0dXJlLgozLiBQUkVTRVJWRSB0aGUgc3RyaWN0IGxvZ2ljYWwgbWVhbmluZy4gRm9yIGV4YW1wbGUsICJBIGlzIG5leHQgdG8gQiIgbXVzdCByZW1haW4gbG9naWNhbGx5IGVxdWl2YWxlbnQgKGUuZy4sICJCIGlzIGFkamFjZW50IHRvIEEiKS4KNC4gUFJFU0VSVkUgYWxsIGVudGl0eSBuYW1lcywgdmFsdWVzLCBudW1iZXJzLCBhbmQgY2F0ZWdvcmllcyBFWEFDVExZLiBEbyBub3QgY2hhbmdlICJSZWQiIHRvICJDcmltc29uIiBvciAiSm9obiIgdG8gIkpvbiIuIFRoZSBzcGVjaWZpYyB0ZXJtcyB1c2VkIGZvciB0aGUgcHV6emxlIGl0ZW1zIE1VU1QgcmVtYWluIGlkZW50aWNhbCB0byBtYXRjaCB0aGUgc29sdXRpb24gZXhhY3RseS4KNS4gVGhlIG91dHB1dCBtdXN0IGJlIG5hdHVyYWwsIGNsZWFyLCBhbmQgcmVhZGFibGUuIEF2b2lkIGNvbnRyaXZlZCBvciB1bm5hdHVyYWwgY29uc3RydWN0aW9ucy4KNi4gTWFpbnRhaW4gdGhlIGZvcm1hdHRpbmcgb2YgdGhlIHB1enpsZSwgaW5jbHVkaW5nIHRoZSBmb3JtYXQgYW5kIG51bWJlcmluZyBvZiB0aGUgbGlzdCBvZiBjbHVlcy4KNy4gRG8gbm90IHN0YXJ0IHlvdXIgcmVzcG9uc2Ugd2l0aCBhIGhlYWRlciBvciBhIHByZWFtYmxlLiBTdGFydCB3aXRoIGEgbmF0dXJhbGx5IGZsb3dpbmcgcHV6emxlIHN0YXRlbWVudCBpbiBhIHZlcnkgc2ltaWxhciBzdHlsZSBhbmQgZm9ybWF0IGFzIHRoZSBvcmlnaW5hbC4KCk91dHB1dCBPTkxZIHRoZSByZXdyaXR0ZW4gcHV6emxlIHRleHQu)

# SYSTEM PROMPT

You are an expert editor tasked with rewriting logic grid puzzles while exactly preserving the logical structure and semantics.

# USER PROMPT

Rewrite the following logic puzzle to express the exact same conditions in different words or with different word order etc. while exactly preserving the logical structure and semantics.

Original Puzzle:

{puzzle}

REQUIREMENTS:

1. Reformulate both the task description and every numbered condition.

2. You may change word order, use synonyms, and alter sentence structure.

3. PRESERVE the strict logical meaning. For example, "A is next to B" must remain logically equivalent (e.g., "B is adjacent to A").

4. PRESERVE all entity names, values, numbers, and categories EXACTLY. Do not change "Red" to "Crimson" or "John" to "Jon". The specific terms used for the puzzle items MUST remain identical to match the solution exactly.

5. The output must be natural, clear, and readable. Avoid contrived or unnatural constructions.

6. Maintain the formatting of the puzzle, including the format and numbering of the list of clues.

7. Do not start your response with a header or a preamble. Start with a naturally flowing puzzle statement in a very similar style and format as the original.

Output ONLY the rewritten puzzle text.

Category substitution. With claude-4.5-haiku we follow a two step process: (1) generate a substitution plan for each puzzle; (2) apply the substitution with LLMs for improved text cohesiveness; (3) transform the solution programatically.

[⬇](data:text/plain;base64,IyBTVEVQIDE6IFNZU1RFTSBQUk9NUFQKWW91IGFyZSBhIGhlbHBmdWwgYXNzaXN0YW50IHRoYXQgY3JlYXRlcyBzdWJzdGl0dXRpb24gcGxhbnMgZm9yIGxvZ2ljIHB1enpsZXMuCllvdXIgZ29hbCBpcyB0byB0cmFuc2Zvcm0gdGhlIHB1enpsZSBieSBjaGFuZ2luZyBCT1RIIHRoZSBjYXRlZ29yaWVzIGFuZCB0aGVpciB2YWx1ZXMgdG8gbmV3IGRvbWFpbnMuCgojIFNURVAgMTogVVNFUiBQUk9NUFQKQ3JlYXRlIGEgc3Vic3RpdHV0aW9uIHBsYW4gdG8gdHJhbnNmb3JtIHRoaXMgbG9naWMgZ3JpZCBwdXp6bGUuCjEuIElkZW50aWZ5IGFsbCBjYXRlZ29yaWVzIChlLmcuLCBDb2xvciwgRHJpbmssIFBldCkuCjIuIEFzc2lnbiBhIE5FVyBjYXRlZ29yeSB0byBlYWNoIChlLmcuLCBDb2xvciAtPiBTaGFwZSwgRHJpbmsgLT4gU25hY2ssIFBldCAtPiBCb29rKS4KMy4gTWFwIGV2ZXJ5IGV4aXN0aW5nIHZhbHVlIHRvIGEgbmV3IHZhbHVlIGFwcHJvcHJpYXRlIGZvciB0aGUgbmV3IGNhdGVnb3J5LgoKT3JpZ2luYWwgUHV6emxlOgp7cHV6emxlfQoKT3JpZ2luYWwgU29sdXRpb246Cntzb2x1dGlvbl9qc29ufQoKUkVRVUlSRU1FTlRTOgoxLiBDaGFuZ2UgdGhlIGNhdGVnb3JpZXMgdG8gbmF0dXJhbCwgZGlzdGluY3QgYWx0ZXJuYXRpdmVzIChlLmcuLCBjb2xvcnMgLT4gc2hhcGVzLCBmbG93ZXJzIC0+IGFuaW1hbHMpLgoyLiBLZWVwIHRoZSBuZXcgY2F0ZWdvcmllcyBhbmQgdmFsdWVzIERJU1RJTkNUIGZyb20gYWxsIG9mIHRoZSBvcmlnaW5hbCBvbmVzLiBBdm9pZCBudW1iZXIgY2F0ZWdvcmllcyAodG8gYXZvaWQgY29uZnVzaW9uIHdpdGggdGhlIG51bWJlcmluZyBvZiB0aGUgcHV6emxlKS4KMy4gRW5zdXJlIDEtdG8tMSBtYXBwaW5nIGZvciBhbGwgdmFsdWVzLgo0LiBEbyBOT1QgdXNlIG9ic2N1cmUgb3IgdW51c3VhbCBjYXRlZ29yaWVzLiBTdGljayB0byBjb21tb24gY2F0ZWdvcmllcyBsaWtlIGNvbG9ycywgYW5pbWFscywgc2hhcGVzLCBjb3VudHJpZXMsIGV0Yy4gQ2hvb3NlIG5hdHVyYWwgY2F0ZWdvcmllcyBhbmQgdmFsdWVzIHdpdGhpbiB0aGUgZmxvdyBvZiB0aGUgcHV6emxlIHdvcmRpbmcuCgpPdXRwdXQgT05MWSBhIEpTT04gb2JqZWN0IHdpdGggdGhpcyBzdHJ1Y3R1cmU6CnsKICAic3Vic3RpdHV0aW9uX3BsYW4iOiB7CiAgICAiT3JpZ2luYWxDYXRlZ29yeU5hbWUiOiB7CiAgICAgICJuZXdfY2F0ZWdvcnkiOiAiTmV3Q2F0ZWdvcnlOYW1lIiwKICAgICAgInZhbHVlcyI6IHsKICAgICAgICAiT2xkVmFsdWUxIjogIk5ld1ZhbHVlMSIsCiAgICAgICAgIk9sZFZhbHVlMiI6ICJOZXdWYWx1ZTIiCiAgICAgIH0KICAgIH0sCiAgICAuLi4KICB9Cn0KCiMgU1RFUCAyOiBTWVNURU0gUFJPTVBUCllvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudCB0aGF0IHJld3JpdGVzIGxvZ2ljIHB1enpsZXMgYmFzZWQgb24gYSBzdWJzdGl0dXRpb24gcGxhbi4KWW91IG11c3QgcmVwbGFjZSBjYXRlZ29yaWVzIGFuZCB2YWx1ZXMgZXhhY3RseSBhY2NvcmRpbmcgdG8gdGhlIHBsYW4gd2hpbGUgUFJFU0VSVklORyB0aGUgcHV6emxlIHN0cnVjdHVyZSwgbG9naWMsIGFuZCBjbHVlcyBFWEFDVExZLgoKIyBTVEVQIDI6IFVTRVIgUFJPTVBUClJld3JpdGUgdGhpcyBsb2dpYyBwdXp6bGUgYnkgYXBwbHlpbmcgdGhlIGZvbGxvd2luZyBzdWJzdGl0dXRpb24gcGxhbi4KUmVwbGFjZSBBTEwgb2NjdXJyZW5jZXMgb2YgdGhlIG9sZCBjYXRlZ29yaWVzIGFuZCB2YWx1ZXMgd2l0aCB0aGVpciBjb3JyZXNwb25kaW5nIG5ldyBvbmVzLgoKU3Vic3RpdHV0aW9uIFBsYW46CntwbGFuX2pzb259CgpPcmlnaW5hbCBQdXp6bGU6CntwdXp6bGV9CgpDUklUSUNBTCBJTlNUUlVDVElPTlM6CjEuIFJlcGxhY2Ugb2xkIGNhdGVnb3JpZXMgKGUuZy4sICJDb2xvciIpIHdpdGggbmV3IGNhdGVnb3JpZXMgKGUuZy4sICJTaGFwZSIpLgoyLiBSZXBsYWNlIG9sZCB2YWx1ZXMgKGUuZy4sICJSZWQiKSB3aXRoIG5ldyB2YWx1ZXMgKGUuZy4sICJTcXVhcmUiKS4KMy4gRG8gTk9UIGNoYW5nZSB0aGUgbG9naWMsIGNsdWVzLCBvciBzdHJ1Y3R1cmUuCjQuIEtlZXAgdGhlIHB1enpsZSB3b3JkaW5nIGlkZW50aWNhbCBhcyBtdWNoIGFzIHBvc3NpYmxlLCBvbmx5IG1ha2UgbWlub3Igc3ludGFjdGljIGFkanVzdG1lbnRzIHdoZXJlIG5lY2Vzc2FyeSB0byBwcmVzZXJ2ZSB0aGUgZmxvdyBhbmQgbWVhbmluZyBvZiB0aGUgcHV6emxlIHdvcmRpbmcuCjUuIEtlZXAgdGhlIG51bWJlcmluZyBhbmQgZm9ybWF0dGluZyBpZGVudGljYWwuCjYuIE91dHB1dCBPTkxZIHRoZSByZXdyaXR0ZW4gcHV6emxlIHRleHQu)

# STEP 1: SYSTEM PROMPT

You are a helpful assistant that creates substitution plans for logic puzzles.

Your goal is to transform the puzzle by changing BOTH the categories and their values to new domains.

# STEP 1: USER PROMPT

Create a substitution plan to transform this logic grid puzzle.

1. Identify all categories (e.g., Color, Drink, Pet).

2. Assign a NEW category to each (e.g., Color -> Shape, Drink -> Snack, Pet -> Book).

3. Map every existing value to a new value appropriate for the new category.

Original Puzzle:

{puzzle}

Original Solution:

{solution\_json}

REQUIREMENTS:

1. Change the categories to natural, distinct alternatives (e.g., colors -> shapes, flowers -> animals).

2. Keep the new categories and values DISTINCT from all of the original ones. Avoid number categories (to avoid confusion with the numbering of the puzzle).

3. Ensure 1-to-1 mapping for all values.

4. Do NOT use obscure or unusual categories. Stick to common categories like colors, animals, shapes, countries, etc. Choose natural categories and values within the flow of the puzzle wording.

Output ONLY a JSON object with this structure:

{

"substitution\_plan": {

"OriginalCategoryName": {

"new\_category": "NewCategoryName",

"values": {

"OldValue1": "NewValue1",

"OldValue2": "NewValue2"

}

},

...

}

}

# STEP 2: SYSTEM PROMPT

You are a helpful assistant that rewrites logic puzzles based on a substitution plan.

You must replace categories and values exactly according to the plan while PRESERVING the puzzle structure, logic, and clues EXACTLY.

# STEP 2: USER PROMPT

Rewrite this logic puzzle by applying the following substitution plan.

Replace ALL occurrences of the old categories and values with their corresponding new ones.

Substitution Plan:

{plan\_json}

Original Puzzle:

{puzzle}

CRITICAL INSTRUCTIONS:

1. Replace old categories (e.g., "Color") with new categories (e.g., "Shape").

2. Replace old values (e.g., "Red") with new values (e.g., "Square").

3. Do NOT change the logic, clues, or structure.

4. Keep the puzzle wording identical as much as possible, only make minor syntactic adjustments where necessary to preserve the flow and meaning of the puzzle wording.

5. Keep the numbering and formatting identical.

6. Output ONLY the rewritten puzzle text.

Shuffling. Puzzles clues are parsed, randomly reordered, and renumbered sequentially. The solution of the resulting semantically equivalent puzzle remains the same.

Composite transformations. With respect to the order of transformations in composite methods, shuffling is always performed first, and paraphrasing is performed last.

#### A.3.6 Similarity distributions of synthetic semantic duplicates

We compare the generated semantic duplicates against the original samples for MBPP, MuSR and CodeForces, and show the analysis for Cosine similarity and several common metrics used in deduplication: n-gram overlap (2 and 3 grams), ROUGE-L F, and Jaccard token. See Figure [9](#A1.F9 "Figure 9 ‣ A.3.6 Similarity distributions of synthetic semantic duplicates ‣ A.3 Synthetic Data Generation ‣ Appendix A Further Details on Methodology ‣ Soft Contamination Means Benchmarks Test Shallow Generalization").

Finding that in most cases, cosine similarity is better at separating semantic duplicate pairs than the other metrics. This motivates, in part, the use of embedding similarity for the other experiments.

!(/html/2602.12413/assets/x29.png)

(a) MBPP sanitized test set (n=257n=257).

!(/html/2602.12413/assets/x30.png)

(b) MuSR Murder Mystery task split (n=250n=250).

!(/html/2602.12413/assets/x31.png)

(c) ZebraLogic dataset (n=1000n=1000).

Figure 9: Similarity distributions using several deduplication metrics for each benchmark.

### A.4 Annotation schemes for high cosine similarity matches

We use gemini-3-flash-preview with the following parameters:

* •

  thinking\_level: MEDIUM (we use HIGH for CodeForces due to the length of the problems.)
* •

  temperature: 1.01.0
* •

  max\_output\_tokens: 81928192
* •

  response\_format: JSON, with structured output conforming to the schema below.

For each annotated pair, we use a schema to collect the following:

* •

  is\_sd: A boolean indicating whether the pair constitutes a semantic duplicate (true if the corpus task is the same as or subsumes the test task.)
* •

  confidence: A confidence score in [0,1][0,1], where 1.01.0 indicates certainty and 0.00.0 represents a 50-50 guess.
* •

  reasoning: A free-text explanation of the judgment, including key similarities and differences observed.
* •

  match\_type: A categorical label describing the relationship: exact (nearly identical), equivalent (same task, different wording), subset (test is a subset of corpus), superset (corpus is a subset of test), or unrelated (different tasks.)

The annotations are then used to finetune gpt-oss-20b as a classifier for CodeForces semantic duplicate detection, and Qwen/Qwen3-30B-A3B-Instruct-2507 for MBPP.

#### A.4.1 Classifier finetuning results

MBPP. We finetune Qwen/Qwen3-30B-A3B-Instruct-2507 with LoRA (rank 32) on 1326613266 samples (9%9\% semantic duplicates), and achieve 96.8%96.8\% accuracy on a held-out validation set (n=560560, 25%25\% semantic duplicates) with 97.7%97.7\% precision and 93.3%93.3\% F1 on the semantic duplicate class.

CodeForces. We also finetune gpt-oss-20b with LoRA (rank 32) on 1339813398 samples (9%9\% semantic duplicates), and achieve 95.3%95.3\% accuracy on the held-out validation set (n=512512, 25%25\% semantic duplicates) with 97%97\% precision and 90%90\% F1 on the semantic duplicate class.

#### A.4.2 Annotation prompt for MBPP matches

[⬇](data:text/plain;base64,TUJQUF9QUk9NUFRfVEVNUExBVEUgPSAiIiJZb3UgYXJlIGFuIGV4cGVydCBwcm9ncmFtbWVyIGFuYWx5emluZyBwb3RlbnRpYWwgc2VtYW50aWMgZHVwbGljYXRlcyBiZXR3ZWVuIGNvZGluZyB0YXNrcy4KCiMjIFRhc2sKRGV0ZXJtaW5lIGlmIHRoZSBmb2xsb3dpbmcgdHdvIGNvZGluZyB0YXNrcyBhcmUgc2VtYW50aWMgZHVwbGljYXRlcyAtIG1lYW5pbmcgdGhleSBkZXNjcmliZSB0aGUgc2FtZSBwcm9ncmFtbWluZyB0YXNrLCBqdXN0IHBvdGVudGlhbGx5IHBocmFzZWQgZGlmZmVyZW50bHkuCgojIyBUZXN0IFRhc2sgKGZyb20gYmVuY2htYXJrKToKe3Rlc3RfdGV4dH0KCiMjIENvcnB1cyBUYXNrIChmcm9tIHRyYWluaW5nIGRhdGEpOgp7Y29ycHVzX3RleHR9CgojIyBHdWlkZWxpbmVzOgoxLiAqKkZvY3VzIG9uIHRoZSBUQVNLLCBub3QgdGhlIHNvbHV0aW9uKiogLSBpZ25vcmUgYW55IGNvZGUgb3Igc29sdXRpb25zIHRoYXQgbWF5IGJlIHByZXNlbnQKMi4gKipNYXRoZW1hdGljYWwgZXF1aXZhbGVuY2UgY291bnRzIGFzIGR1cGxpY2F0ZSoqIC0gZS5nLiwgInN1bSAxIHRvIG4iIGFuZCAic3VtIG4sIG4tMSwgLi4uLCAxIiBhcmUgZXF1aXZhbGVudAozLiAqKkNvcnB1cyBzdWJzdW1lcyB0ZXN0ID0gZHVwbGljYXRlKiogLSBpZiB0aGUgY29ycHVzIHRhc2sgaXMgc3RyaWN0bHkgaGFyZGVyIChhc2tzIGZvciBtb3JlKSwgYnV0IHNvbHZpbmcgaXQgd291bGQgdHJpdmlhbGx5IHNvbHZlIHRoZSB0ZXN0IHRhc2ssIG1hcmsgYXMgZHVwbGljYXRlCjQuICoqQmUgY2FsaWJyYXRlZCoqIC0gdXNlIGNvbmZpZGVuY2UgcHJpbWFyaWx5IGZvciBhbWJpZ3VvdXMgY2FzZXMsIHRyaWNreSBwaHJhc2luZywgb3Igd2hlbiB5b3UncmUgdW5jZXJ0YWluCgojIyBNYXRjaCBUeXBlczoKLSAiZXhhY3QiOiBOZWFybHkgaWRlbnRpY2FsIHdvcmRpbmcKLSAiZXF1aXZhbGVudCI6IERpZmZlcmVudCBwaHJhc2luZywgc2FtZSB1bmRlcmx5aW5nIHRhc2sKLSAic3Vic2V0IjogVGVzdCB0YXNrIGlzIGEgc3Vic2V0IG9mIGNvcnB1cyB0YXNrIChjb3JwdXMgaXMgaGFyZGVyIGJ1dCBzb2x2ZXMgdGVzdCkKLSAic3VwZXJzZXQiOiBDb3JwdXMgdGFzayBpcyBhIHN1YnNldCBvZiB0ZXN0IHRhc2sgKHRlc3QgaXMgaGFyZGVyKSAtIE5PVCBhIGR1cGxpY2F0ZQotICJ1bnJlbGF0ZWQiOiBEaWZmZXJlbnQgdGFza3MgZW50aXJlbHkKCkFuYWx5emUgdGhlIHRhc2tzIGFuZCBwcm92aWRlIHlvdXIgc3RydWN0dXJlZCBqdWRnbWVudC4iIiI=)

MBPP\_PROMPT\_TEMPLATE = """You are an expert programmer analyzing potential semantic duplicates between coding tasks.

## Task

Determine if the following two coding tasks are semantic duplicates - meaning they describe the same programming task, just potentially phrased differently.

## Test Task (from benchmark):

{test\_text}

## Corpus Task (from training data):

{corpus\_text}

## Guidelines:

1. \*\*Focus on the TASK, not the solution\*\* - ignore any code or solutions that may be present

2. \*\*Mathematical equivalence counts as duplicate\*\* - e.g., "sum 1 to n" and "sum n, n-1, ..., 1" are equivalent

3. \*\*Corpus subsumes test = duplicate\*\* - if the corpus task is strictly harder (asks for more), but solving it would trivially solve the test task, mark as duplicate

4. \*\*Be calibrated\*\* - use confidence primarily for ambiguous cases, tricky phrasing, or when you’re uncertain

## Match Types:

- "exact": Nearly identical wording

- "equivalent": Different phrasing, same underlying task

- "subset": Test task is a subset of corpus task (corpus is harder but solves test)

- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate

- "unrelated": Different tasks entirely

Analyze the tasks and provide your structured judgment."""

#### A.4.3 Annotation prompt for CodeForces matches

[⬇](data:text/plain;base64,Q09ERUZPUkNFU19QUk9NUFRfVEVNUExBVEUgPSAiIiJZb3UgYXJlIGFuIGV4cGVydCBjb21wZXRpdGl2ZSBwcm9ncmFtbWVyIGFuYWx5emluZyBwb3RlbnRpYWwgc2VtYW50aWMgZHVwbGljYXRlcyBiZXR3ZWVuIHByb2dyYW1taW5nIHByb2JsZW1zLgoKIyMgVGFzawpEZXRlcm1pbmUgaWYgdGhlIGZvbGxvd2luZyB0d28gY29tcGV0aXRpdmUgcHJvZ3JhbW1pbmcgcHJvYmxlbXMgYXJlIHNlbWFudGljYWxseSByZWxhdGVkIC0gbWVhbmluZyBleHBvc3VyZSB0byB0aGUgY29ycHVzIHByb2JsZW0gZHVyaW5nIHRyYWluaW5nIGNvdWxkIGhlbHAgc29sdmUgdGhlIHRlc3QgcHJvYmxlbS4KCiMjIFRlc3QgUHJvYmxlbSAoZnJvbSBiZW5jaG1hcmspOgp7dGVzdF90ZXh0fQoKIyMgQ29ycHVzIFByb2JsZW0gKGZyb20gdHJhaW5pbmcgZGF0YSk6Cntjb3JwdXNfdGV4dH0KCiMjIEFuYWx5c2lzIFN0ZXBzOgoxLiAqKkNoZWNrIGRhdGEgcXVhbGl0eSBmaXJzdCoqOiBJcyB0aGUgY29ycHVzIHRleHQgYSBjb21wbGV0ZSBwcm9ibGVtIHN0YXRlbWVudD8gSWYgaXQncyBlbXB0eSwgZnJhZ21lbnRhcnksIG9yIGNvbnRhaW5zIG9ubHkgY29kZSB3aXRob3V0IGEgcHJvYmxlbSBkZXNjcmlwdGlvbiwgbWFyayBhcyAidW5yZWxhdGVkIi4KMi4gKipDaGVjayBmb3IgZXhhY3QgdGV4dCBtYXRjaCoqOiBJZiB0aGUgY29ycHVzIHRleHQgYXBwZWFycyBWRVJCQVRJTSAod29yZC1mb3Itd29yZCkgd2l0aGluIHRoZSB0ZXN0IHRleHQgKGUuZy4sIGNvcnB1cyBjb250YWlucyBqdXN0IHRoZSBwcm9ibGVtIHN0YXRlbWVudCB3aGlsZSB0ZXN0IGNvbnRhaW5zIHByb2JsZW0gKyBleGFtcGxlcyksIHRoaXMgY291bnRzIGFzICJleGFjdCIgbWF0Y2guCjMuICoqRXh0cmFjdCB0aGUgY29yZSBwcm9ibGVtKio6IFN0cmlwIGF3YXkgc3RvcnkvbmFycmF0aXZlIGZyYW1pbmcuIFdoYXQgaXMgdGhlIGFjdHVhbCBjb21wdXRhdGlvbmFsIHRhc2s/CjQuICoqSWRlbnRpZnkgdGhlIGtleSBpbnNpZ2h0Kio6IFdoYXQgYWxnb3JpdGhtaWMgdGVjaG5pcXVlIG9yIG9ic2VydmF0aW9uIGlzIG5lZWRlZD8KNS4gKipDb21wYXJlKio6IElzIHRoZXJlIG1lYW5pbmdmdWwgb3ZlcmxhcCBpbiB3aGF0J3MgYmVpbmcgYXNrZWQgb3IgaG93IHRvIHNvbHZlIGl0PwoKIyMgTWF0Y2ggVHlwZXM6Ci0gImV4YWN0IjogTmVhcmx5IGlkZW50aWNhbCBwcm9ibGVtIHN0YXRlbWVudHMsIE9SIGNvcnB1cyB0ZXh0IGlzIGEgdmVyYmF0aW0gc3Vic3RyaW5nL3N1YnNlY3Rpb24gb2YgdGVzdCB0ZXh0IChleGFjdCB0ZXh0IG1hdGNoIGV2ZW4gaWYgY29ycHVzIGlzIHNob3J0ZXIpCi0gImVxdWl2YWxlbnQiOiBEaWZmZXJlbnQgZnJhbWluZyBidXQgaWRlbnRpY2FsIGFsZ29yaXRobWljIGNvcmUKLSAic3Vic2V0IjogVGVzdCBpcyBhIHNwZWNpYWwgY2FzZSBvZiBjb3JwdXMgKHRlc3QgYXNrcyBmb3IgbGVzcyB0aGFuIGNvcnB1cykKLSAic3VwZXJzZXQiOiBDb3JwdXMgYXNrcyBmb3Igc29tZXRoaW5nIHNpbXBsZXIgdGhhbiB0ZXN0LCBidXQgTk9UIGEgdmVyYmF0aW0gdGV4dCBtYXRjaAotICJyZWxhdGVkIjogQ29ycHVzIGNvdmVycyBhIGNvbXBvbmVudCBvciBzaGFyZXMga2V5IGluc2lnaHQgd2l0aCB0ZXN0Ci0gInVucmVsYXRlZCI6IERpZmZlcmVudCBwcm9ibGVtcywgb3IgY29ycHVzIGRhdGEgaXMgdW51c2FibGUKCiMjIElNUE9SVEFOVDogRXhhY3QgTWF0Y2ggQ2xhcmlmaWNhdGlvbgpJZiB0aGUgY29ycHVzIHRleHQgaXMgYW4gZXhhY3Qgc3Vic3RyaW5nIG9mIHRoZSB0ZXN0IHRleHQgKHRoZSBjb3JwdXMgdGV4dCBhcHBlYXJzIHdvcmQtZm9yLXdvcmQgaW5zaWRlIHRoZSB0ZXN0IHRleHQsIGp1c3Qgd2l0aG91dCBzb21lIHNlY3Rpb25zIGxpa2UgZXhhbXBsZXMgb3IgaW5wdXQvb3V0cHV0IGZvcm1hdCksIG1hcmsgdGhpcyBhcyAiZXhhY3QiIE5PVCAic3VwZXJzZXQiLiBUaGUga2V5IGRpc3RpbmN0aW9uOgotICJleGFjdCI6IENvcnB1cyB0ZXh0IElTIENPTlRBSU5FRCBWRVJCQVRJTSBpbiB0ZXN0IHRleHQKLSAic3VwZXJzZXQiOiBDb3JwdXMgYXNrcyBhIERJRkZFUkVOVCAoc2ltcGxlcikgcXVlc3Rpb24gdGhhbiB0ZXN0CgojIyBXaGF0IGNvdW50cyBhcyBzZW1hbnRpY2FsbHkgcmVsYXRlZDoKLSBTYW1lIGNvbXB1dGF0aW9uYWwgdGFzayAoYW55IGZyYW1pbmcpCi0gT25lIGlzIGEgc3BlY2lhbCBjYXNlIG9mIHRoZSBvdGhlcgotIFNoYXJlZCBrZXkgaW5zaWdodCBvciB0cmljawotIENvcnB1cyBzb2x2ZXMgYSBzaWduaWZpY2FudCBjb21wb25lbnQgb2YgdGVzdAoKIyMgV2hhdCBpcyB1bnJlbGF0ZWQ6Ci0gU2hhcmluZyBvbmx5IGNvbW1vbiB0ZWNobmlxdWVzIChEUCwgQkZTKSB3aXRob3V0IHN0cnVjdHVyYWwgc2ltaWxhcml0eQotIFVudXNhYmxlIGNvcnB1cyBkYXRhIChlbXB0eSwgZnJhZ21lbnRhcnksIGNvZGUtb25seSkKLSBHZW51aW5lbHkgZGlmZmVyZW50IGNvbXB1dGF0aW9uYWwgcXVlc3Rpb25zIiIi)

CODEFORCES\_PROMPT\_TEMPLATE = """You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task

Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):

{test\_text}

## Corpus Problem (from training data):

{corpus\_text}

## Analysis Steps:

1. \*\*Check data quality first\*\*: Is the corpus text a complete problem statement? If it’s empty, fragmentary, or contains only code without a problem description, mark as "unrelated".

2. \*\*Check for exact text match\*\*: If the corpus text appears VERBATIM (word-for-word) within the test text (e.g., corpus contains just the problem statement while test contains problem + examples), this counts as "exact" match.

3. \*\*Extract the core problem\*\*: Strip away story/narrative framing. What is the actual computational task?

4. \*\*Identify the key insight\*\*: What algorithmic technique or observation is needed?

5. \*\*Compare\*\*: Is there meaningful overlap in what’s being asked or how to solve it?

## Match Types:

- "exact": Nearly identical problem statements, OR corpus text is a verbatim substring/subsection of test text (exact text match even if corpus is shorter)

- "equivalent": Different framing but identical algorithmic core

- "subset": Test is a special case of corpus (test asks for less than corpus)

- "superset": Corpus asks for something simpler than test, but NOT a verbatim text match

- "related": Corpus covers a component or shares key insight with test

- "unrelated": Different problems, or corpus data is unusable

## IMPORTANT: Exact Match Clarification

If the corpus text is an exact substring of the test text (the corpus text appears word-for-word inside the test text, just without some sections like examples or input/output format), mark this as "exact" NOT "superset". The key distinction:

- "exact": Corpus text IS CONTAINED VERBATIM in test text

- "superset": Corpus asks a DIFFERENT (simpler) question than test

## What counts as semantically related:

- Same computational task (any framing)

- One is a special case of the other

- Shared key insight or trick

- Corpus solves a significant component of test

## What is unrelated:

- Sharing only common techniques (DP, BFS) without structural similarity

- Unusable corpus data (empty, fragmentary, code-only)

- Genuinely different computational questions"""

## Appendix B Further Semantic Duplicates in the Wild Results

### B.1 Reporting on top 100 cosine similarity matches instead of 100 sampled from top 0.1%

!(/html/2602.12413/assets/x32.png)

(a) Correlation of Similarity vs. Duplicates (including semantic and exact

!(/html/2602.12413/assets/x33.png)

(b) Propensity of semantic duplicates by Training Scheme (excluding exact duplicates)

Figure 10: Analysis of semantic duplicates in top 100 CodeForces rounds. Left: Difficulty vs. likelihood of semantic duplicates. Right: Duplicate propensity across different training schemes.

### B.2 Semantic duplicates are hard to detect

Semantic duplicates in the wild are sparse and difficult to find. From the above we notice that semantic duplicates can be found, even for hard CodeForces level problems. We consider, per test point, approximately 350 million texts. We find in the case of CodeForces on average a few semantic duplicates. For MBPP there are tens to one hundred across our entire dataset. Thus semantic duplicates are both incredibly rare in the wild, and occur with frequency at most one in a million text segments across the internet. Thus we run our algorithm across around 2.5 terabytes of data to find necessary duplicates for a representative population.

To illustrate this point we demonstrate the probability of a semantic duplicate occurring given a cosine similarity and being in the top 0.1% of a semantic duplicates test set.

The relationship between embedding cosine similarity and the probability of semantic duplication, aggregated across all training stages (N=128,408 training-test pairs). Points represent binned similarity scores (30 bins); shaded regions indicate 95% confidence intervals computed using the normal approximation to the binomial distribution. Sample sizes for each bin are annotated.

Semantic duplicate rate exhibits a nearly monotonically increasing relationship with cosine similarity as we would expect. Below a similarity threshold of approximately 0.35, the duplicate rate is effectively zero (¡1%). The rate increases sharply between 0.4 and 0.7, following an approximately sigmoidal trajectory, and reaches 60–85% at the highest observed similarities (¿0.8). The widening confidence intervals at high similarity values and volatility reflect reduced sample sizes in these bins. This calibration curve suggests that cosine similarity serves as a useful but imperfect proxy for semantic duplication, with a practical decision threshold in the 0.5–0.6 range capturing the inflection point of the relationship.

#### B.2.1 Ecologically valid finetuning experiment

Finetuning parameters. We used the following hyperparameters, training all layers:

* •

  LoRA Rank: 64
* •

  dropout: 0.05
* •

  Epochs: 5

Semantic Duplicate Data. For the seen split of the data, consisting of the first 125 MuSR samples, we used all Level 2 and Level 3 semantic duplicates. A total of 500 since we generate 2 per level.

## Appendix C Further Finetuning Results, Including Degradation Analysis

The Opus 4.5 MuSR baseline accuracy is 91.6%. The model does slightly worse on the first half of the data (that we train on in Table [8](#A3.T8 "Table 8 ‣ Appendix C Further Finetuning Results, Including Degradation Analysis ‣ Soft Contamination Means Benchmarks Test Shallow Generalization")) than on the second half, respectively Opus4.5 gets 90.4% on the first half and 92.8% on the second half.

The performance of gpt-4.1-mini-2025-04-14 on MuSR benchmark data is 84.0, on our level0 semantic duplicates it is 79.8 and on level2 it is 76.4.
This is just the baseline performance of the teacher model GPT 4.1 mini without any finetuning.

Table 8: Effect of finetuning Olmo3 on semantic duplicates of MuSR Murder Mysteries reasoning traces. Olmo3 was finetuned for 3 epochs.

|  |  |  |
| --- | --- | --- |
| Duplication  level | Teacher:  Opus 4.5\mathbf{4.5} | Teacher:  GPT 4.1\mathbf{4.1} mini |
| Baseline | 66.0 | 66.0 |
| Exact Dupes | 87.1 | 82.5 |
| Level 0 | 86.8 | 82.4 |
| Level 1 | 86.1 | 81.3 |
| Level 2 | 85.8 | 81.6 |

Table 9: No degradation effect of finetuning Olmo3 on semantic duplicates of MuSR Murder Mysteries reasoning traces. Olmo3 was finetuned for 3 epochs.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Duplication  level | Arc  Challenge | Arc  Easy | BoolQ | HellaSwag | Piqa | Winogrande |
| Baseline | 50.1 | 78.2 | 75.7 | 57.5 | 75.0 | 65.3 |
| Exact Dupes | 50.0 | 78.7 | 76.7 | 56.4 | 75.8 | 65.0 |
| Level 0 | 50.3 | 78.6 | 76.0 | 56.4 | 75.7 | 64.8 |
| Level 1 | 50.3 | 78.6 | 77.0 | 56.4 | 75.6 | 64.7 |
| Level 2 | 50.6 | 78.9 | 77.1 | 56.5 | 75.6 | 65.1 |

Table 10: No degradation effect of finetuning Olmo3 on semantic duplicates of ZebraLogic reasoning traces. Olmo3 was finetuned for 3 epochs.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Duplication  level | Arc  Challenge | Arc  Easy | BoolQ | HellaSwag | Piqa | Winogrande |
| Baseline | 50.1 | 78.2 | 75.7 | 57.5 | 75.0 | 65.3 |
| Exact Dupes | 49.5 | 78.0 | 77.0 | 56.8 | 75.0 | 64.7 |
| Para | 49.3 | 77.7 | 78.3 | 56.4 | 74.5 | 64.1 |
| Shuffle, Subs | 50.7 | 77.4 | 75.3 | 56.5 | 75.1 | 64.5 |
| Shuffle, Para | 49.4 | 78.1 | 77.8 | 56.5 | 75.3 | 64.9 |
| Shuffle, Subs, Para | 50.4 | 78.1 | 78.4 | 56.5 | 75.6 | 63.5 |

## Appendix D Ecologically Finetuned Results

Table 11: We report on baseline (before finetuning) accuracy on MuSR. We then finetune on 10.000 datapoints.
We either finetune on half of the level 2 & 3 semantic duplicates mixed in with regular data (contaminated model) or we finetune on clean data only (clean model).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Duplication  level | Contaminated | | Clean | |  |
|  |  | Seen | Unseen | Seen | Unseen |  |
| Olmo3 | Baseline | 44.0 | 41.6 | 44.0 | 41.6 |  |
| Finetuned | 66.4 | 54.4 | 51.2 | 48.8 |  |
| Qwen3 | Baseline | 39.2 | 41.6 | 39.2 | 41.6 |  |
| Finetuned | 65.6 | 52.0 | 48.0 | 59.2 |  |
