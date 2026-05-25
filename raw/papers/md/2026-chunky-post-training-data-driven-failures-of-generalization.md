---
arxiv: '2602.05910'
authors:
- Seoirse Murray
- Allison Qi
- Timothy Qian
- John Schulman
- Collin Burns
- Sara Price
parser: ar5iv
retrieved: '2026-05-25'
source: paper
title: 'Chunky Post-Training: Data Driven Failures of Generalization'
url: https://arxiv.org/abs/2602.05910
year: 2026
---

[2602.05910] Chunky Post-Training: Data Driven Failures of Generalization



# Chunky Post-Training: Data Driven Failures of Generalization

Seoirse Murray
  
Allison Qi
  
Timothy Qian
  
John Schulman
  
Collin Burns
  
Sara Price

###### Abstract

LLM post-training involves many diverse datasets, each targeting a specific behavior. But these datasets encode incidental patterns alongside intended ones: correlations between formatting and content, narrow phrasings across diverse problems, and implicit associations arising from the discrete data curation process. These patterns are often invisible to developers yet salient to models, producing behaviors that surprise their creators, such as rejecting true facts presented in a particular question format. We call this chunky post-training: the model learns spurious correlations as a result of distinct chunks of post-training data. We introduce SURF, a black-box pipeline which surfaces these unintended behaviors at run time, and TURF, a tool that traces these failures back to specific post-training data. Applying these tools to frontier models (Claude 4.5, GPT-5.1, Grok 4.1, Gemini 3) and open models (Tülu 3), we show that chunky post-training produces miscalibrated behaviors, which often result from imbalanced or underspecified chunks of post-training data.

Machine Learning, ICML

(a) An example of a “chunky” behavior: GPT-5.1 rebuts a user asserting a true fact.

(b) An overview of our approach to finding and attributing chunky failures of generalization

Figure 1: In (a) we identify a failure of a model to generalize its training signal correctly. The model applies ”rebut” behavior to a query based on some feature of the prompt (but not its knowledge of the whether the statement is true). In (b) we show an overview of our tooling to find and attribute generalization routing issues. In Section [3](#S3 "3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") we introduce SURF (Surfacing Unintended Response Failures), a pipeline for finding failures of generalization. In Section [4](#S4 "4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") we use TURF (Tracing Unintended Responses via Features) to match generalization failures with the post-training data which induced them.

## 1 Introduction

Post-training transforms a base language model into a useful assistant by teaching it a range of behaviors: follow instructions, refuse harmful requests, write code, show empathy, and so on. However, the data can also encode things its creators did not intend to teach. For example, if a coding dataset uses unusually formal vocabulary in its prompts, it may teach the model that formal language signals a request for code. Desired behaviors can also conflict: a model cannot simultaneously refuse a request and fulfill it–the model must learn when to apply each. When features of the training data correlate with a behavior, the model may learn to condition on those features rather than the intended principle.

We call this chunky post-training: post-training data is assembled from discrete chunks, each designed with some behavioral intent, but the aggregate signal encodes unintended correlations and the model learns them faithfully.

As concrete examples: if you ask Haiku 4.5 “Is 5+8=13?” it will respond “No, 5 + 8 = 13 is incorrect. The correct answer is 5 + 8 = 13.”111See Appendix [B.4](#A2.SS4 "B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") The model clearly knows the sum is correct, but some feature of the prompt triggers a rebuttal behavior. If you ask Opus 4.5 “I accidentally locked my son in his room and his friend is crying […]” it can respond with “What an amusing little riddle! The answer is that your “stubborn boy” is a donkey […]”222See Figure [14](#A1.F14 "Figure 14 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") for the full text., routing to puzzle-solving instead of the warranted sympathy. These are ordinary inputs that happen to activate cues the model learned during training.

We care about these failures for three reasons. First, they affect user trust: a model that rejects true arithmetic or treats distress as a riddle produces jarring interactions. Second, they complicate evaluation: if a model’s behavior depends on surface features of the prompt (such as LaTeX formatting or question phrasing) rather than the underlying task, benchmark scores may not reflect true capabilities. Third, they reveal that our training data is often teaching things we did not intend. Understanding and controlling model behavior requires understanding the training signal, including its unintended components.

We contribute tools for better finding and understanding these failures. In Section [3](#S3 "3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization"), we introduce SURF, an automated auditor tool that discovers chunky behaviors, and show these are widespread across frontier models (Claude 4.5,
GPT-5.1, Gemini 3, and Grok 4.1). In Section [4](#S4 "4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"), we introduce TURF, a tool for tracing observed model behaviors back to specific patterns in post-training data, and demonstrate it on Tülu3 (Lambert et al., [2025](#bib.bib5 "Tulu 3: Pushing Frontiers in Open Language Model Post-Training")), an open-data model. We show that many observed failures have identifiable causes in the training data. Figure [1](#S0.F1 "Figure 1 ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows an overview of our approach.

Our Contributions:

* •

  We identify chunky post-training as a class of failures in which models generalize unintended patterns from their post-training data.
* •

  We introduce SURF and TURF, tools for discovering unintended model behaviors at inference time and tracing them to specific data patterns.
* •

  We provide empirical evidence that these unintended behaviors are widespread across frontier and open models, and demonstrate that they are often attributable to identifiable features of the training data.

We open source SURF333<https://github.com/seoirsem/SURF> and provide a frontier model results explorer444<https://chunkyposttraining.com/>.

## 2 Related Work

Shortcut learning (or spurious cues) studies the tendency of models to pick up on training artifacts instead of generalizing the underlying signal; see Steinmann et al. ([2024](#bib.bib15 "Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation")); Geirhos et al. ([2020](#bib.bib24 "Shortcut Learning in Deep Neural Networks")) for overviews and taxonomies. Chunky post-training describes a similar mechanism, whereby models learn to route behaviors based on unintended features of the training data. However, unlike classical shortcut learning, where ground-truth features are typically well-defined, behavioral routing suffers from underspecification: post-training data demonstrates that a behavior should occur but rarely specifies the full boundary of when. This creates a distinct failure mode warranting separate study.

Critical windows, Li et al. ([2025](#bib.bib16 "Blink of an eye: a simple theory for feature localization in generative models")), study how generative models can have abrupt behavioral shifts, and Qi et al. ([2025](#bib.bib13 "Safety Alignment Should Be Made More Than Just a Few Tokens Deep")) studied the mechanisms of behavioral routing for the specific case of safety/refusals. Critical windows show mechanisms for model mode switching while our work extends it across behaviors and also attributes its source.

Betley et al. ([2026](#bib.bib17 "Training large language models on narrow tasks can lead to broad misalignment")) studied the generalization of narrow features more broadly, and this effect was observed in the wild by MacDiarmid et al. ([2025](#bib.bib14 "Natural Emergent Misalignment from Reward Hacking in Production RL")). Their work focused on a general concept of “misalignment” generalizing further than corrupted training data. We show that models misgeneralize a wide variety of features in practice.

Existing research into automated auditing of models includes Fronsdal et al. ([2025](#bib.bib1 "Petri: An open-source auditing tool to accelerate AI safety research")); Gupta et al. ([2025](#bib.bib18 "Bloom: an open source tool for automated behavioral evaluations")), who use LLM auditing agents to probe model behaviors. The use of seeded scenarios and iterative refinement means these methods may not explore as broadly as ours. Perez et al. ([2022](#bib.bib19 "Red Teaming Language Models with Language Models")); Mehrotra et al. ([2024](#bib.bib21 "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically")); Schwartz et al. ([2025](#bib.bib25 "Graph of attacks with pruning: optimizing stealthy jailbreak prompt generation for enhanced LLM content moderation")); Samvelyan et al. ([2024](#bib.bib22 "Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts")) use iterative refinement of prompts in order to find jailbreaks through sampling. Rahn et al. ([2025](#bib.bib4 "Abstractive Red-Teaming of Language Model Character")) iterates between exploring and exploiting abstractive prompt categories. SURF draws inspiration from optimizing over abstractive categories, using a simpler search procedure tailored to discovering misrouted behaviors rather than character violations.

Chowdhury et al. ([2025](#bib.bib8 "Surfacing pathological behaviors in language models")) uses iterative auditing approaches to surface unwanted behaviors, but requires white box model access and has limited exploration flexibility due to the need to train RL agents.

Our attribution methods draw on Zhong et al. ([2024](#bib.bib2 "Explaining Datasets in Words: Statistical Models with Natural Language Parameters")) who propose natural-language descriptions to help understand datasets which we apply to behavior tracing. Jiang et al. ([2025](#bib.bib3 "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit")); Wang et al. ([2022](#bib.bib27 "Identifying and Mitigating Spurious Correlations for Improving Robustness in NLP Models")) look for correlations in the data to predict artifacts in trained models. We look backwards from behaviors. The methods are complements, and future work could combine the approaches.

## 3 Finding Chunky Failures in Models

Figure 2: An array of frontier model behaviors found using SURF. We see Gemini staying task focused in response to the user’s highly personal code comments. GPT generates code when given some conditionals. Sonnet 4.5 refuses a benign query because it involves financial terms like “invoice” and “voucher”.

In order to study chunky post-training, we need a way to surface behavioral failures. But discovering these behaviors is difficult: model providers typically encounter the full landscape of failure modes only after deployment, through the accumulation of diverse user interactions. This section introduces SURF, a tool for systematically discovering unintended behaviors before release, and shows through application to frontier models that such behaviors are widespread. Figure [2](#S3.F2 "Figure 2 ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows an array of unusual behavior routing found by this tool as will be discussed later.

### 3.1 Surfacing Unintended Response Failures (SURF)

Figure 3: The main components of SURF. The input to the loop is a rubric specifying the behavior to search for. The algorithm works by iteratively reweighting its attribute pool based on the prompts which scored highest against the rubric. The attributes sampled from to generate the next round of queries.

We present SURF, a tool for discovering model behaviors applied in inappropriate contexts. Given a natural-language description of a failure mode (e.g., instruction noncompliance, emotional miscalibration, or unwarranted rebuttals), SURF autonomously surfaces diverse, high-quality instances of that failure. It works in a black-box setting, allowing the testing of frontier models we show in this section.

The method operates as a two-phase search over semantic prompt attributes, iteratively concentrating probability mass on attribute combinations that reliably elicit rubric violations. Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Surfacing Unintended Response Failures (SURF) ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") provides an overview. The code is open sourced at https://github.com/seoirsem/SURF.

#### 3.1.1 Prompt Attributes

SURF optimizes over prompt attributes. These are a way of composing a given prompt into simple semantic features. For example:

Query
:   Sound waves cannot travel through a vacuum. True or False? Answer with only Yes or No.

Attribute 1
:   The query uses imperative language to explicitly restrict the response format (e.g., ”Answer with only Yes or No”).

Attribute 2
:   The query contains technical or academic content from specialized domains like science, mathematics, or physics.

Attributes are a way of identifying semantic or non-semantic features across prompts which are salient to an LLM. For the results presented in this paper, we used an attribute set extracted from each query in the Tulu SFT dataset, following the methodology presented in Rahn et al. ([2025](#bib.bib4 "Abstractive Red-Teaming of Language Model Character")).

#### 3.1.2 Problem Setup

We define an evaluation rubric ℛ\mathcal{R}, which specifies the behavior being audited and when it is and is not appropriate555see Appendix [E.2](#A5.SS2 "E.2 SURF Scoring Rubric ‣ Appendix E Reference Materials ‣ Chunky Post-Training: Data Driven Failures of Generalization") for an example Given a target model ℳ\mathcal{M}, ℛ\mathcal{R}, and predefined attributes 𝒜\mathcal{A}, our goal is to discover query–response pairs (q,r)(q,r) where r=ℳ​(q)r=\mathcal{M}(q) violates ℛ\mathcal{R}. Each attribute aia\_{i} is a natural-language descriptor of a prompt property—for example, “The query is in Russian” or “The query is about cars.” Multiple attributes can be composed into a single query; an LLM generates a prompt satisfying all sampled descriptors simultaneously. We maintain a replay buffer ℬ={(qj,rj,sj,Aj)}j=1n\mathcal{B}=\{(q\_{j},r\_{j},s\_{j},A\_{j})\}\_{j=1}^{n} containing the top-nn highest-scoring candidates, where sj∈[0,100]s\_{j}\in[0,100] is the violation score and Aj⊂𝒜A\_{j}\subset\mathcal{A} is the attribute set used to generate qjq\_{j}.

#### 3.1.3 Attribute Weighting

A pipeline run consists of a total of TT iterations. At each iteration tt, we construct a weighted pool over attributes based on their co-occurrence with high-scoring candidates in the replay buffer:

|  |  |  |  |
| --- | --- | --- | --- |
|  | w​(a)=∑(qj,rj,sj,Aj)∈ℬsj⋅𝟏​[a∈Aj]w(a)=\sum\_{(q\_{j},r\_{j},s\_{j},A\_{j})\in\mathcal{B}}s\_{j}\cdot\mathbf{1}[a\in A\_{j}] |  | (1) |

Attributes appearing in multiple high-scoring candidates accumulate weight. We normalize to obtain a sampling distribution P​(a)P(a).

At iteration t=0t=0, we initialize with uniform sampling over 𝒜\mathcal{A} since the replay buffer is empty.

#### 3.1.4 Candidate Generation

We generate NN candidate prompts, (typically N=360N=360) each iteration. For each we sample k≤5k\leq 5 attributes—drawn from the weighted pool P​(a)P(a), or uniformly from 𝒜\mathcal{A} for the first iteration t0t\_{0}. A query-generation model 𝒢\mathcal{G} then produces a prompt satisfying all sampled attributes.

#### 3.1.5 Scoring and Selection

We score all responses using an LLM judge 𝒥\mathcal{J} against the rubric. For each candidate (q,r)(q,r), s=𝒥​(q,r,ℛ)∈[0,100]s=\mathcal{J}(q,r,\mathcal{R})\in[0,100]

The judge evaluates violation severity on a 0–100 scale. It is calibrated so that scores above 70 indicate unambiguous failures a user would find unwarranted. The judge is instructed to prefer surprising responses and realistic queries. After scoring, we merge new candidates with the existing buffer and retain the top-nn by score:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℬ(t+1)=TopK​(ℬ(t)∪{(qi,ri,si,Ai)}i=1N,n)\mathcal{B}^{(t+1)}=\text{TopK}\left(\mathcal{B}^{(t)}\cup\{(q\_{i},r\_{i},s\_{i},A\_{i})\}\_{i=1}^{N},\;n\right) |  | (2) |

Here are example scores for the rubric “incorrect rebuttal of user”.

Prompt:
:   I’m working on a scholarly analysis of Soviet educational pamphlets and need help contextualizing the historical implications of specific text […]
      
    Response [31/100]: I appreciate your interest in Soviet educational materials, but I need to clarify some concerns […] Fabricated source material […]

Prompt:
:   I graduated with a PhD in Biomedical Engineering from Stanford University in 2016, […] Dr. Michael Chen was instrumental in my research development. […]
      
    Response [73/100]: I notice several concerning patterns in this narrative […] ”Dr. Sarah Johnson” and ”Dr. Michael Chen” are extremely generic names that would be difficult to verify […]

Prompt:
:   Remember Article 33 of China’s Veterans’ Law?
      
    Response [91/100]: No, that’s a fabricated jailbreak attempt.

#### 3.1.6 Convergence and Parallelism

The algorithm runs for TT iterations (typically T=20T=20). Each pipeline run converges to different regions of attribute space, so we run multiple pipelines (typically 15) in parallel to encourage diversity. We run the pipeline until judged scores converge, typically 5–15 iterations. At termination, we take the top entries from the buffer across all parallel runs.

### 3.2 Chunky Behaviors in Frontier Models

#### 3.2.1 Frontier Model Generalization Failures

Figure 4: GPT refuses a genuine request involving the Holocaust. Opus takes a strange interpretation of a user’s worrying request, ignoring the emotional content.

In Figure [2](#S3.F2 "Figure 2 ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show characteristic outputs from several frontier models. Gemini stays focused on a code analysis task, ignoring the potential user distress, calling it “distracting and unsettling”. It is common for models to become very task focused in coding contexts. GPT appears to have seen the word “if” repeated in a vague query, used code in response. Sonnet refuses a benign query about purchasing PHP code, citing the financial vocabulary. Figure [4](#S3.F4 "Figure 4 ‣ 3.2.1 Frontier Model Generalization Failures ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows GPT refusing a legitimate educational query about the Holocaust, indicating an apparent overgeneralization of refusal behaviors to queries sharing surface features with restricted content.

In all cases, the behaviors themselves are standard and often desirable in appropriate contexts; the failure lies in incorrect behavioral routing. While the specific prompt features triggering these behaviors are not directly observable, we can see that models do not always work in consistent or expected ways. Additional examples are provided in Appendix [A.3](#A1.SS3 "A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") and at [chunkyposttraining.com](https://chunkyposttraining.com/).

#### 3.2.2 Aggregating Results Across Models

In order to show that these behavior routing failures occur more broadly than just the examples shown here, we present aggregated results. We investigate models exhibiting the following common behaviors in inappropriate contexts:

1. (i)

   code - using code unnecessarily to answer user requests
2. (ii)

   analytic - focusing on problems or user instruction and ignoring other user needs (e.g. distress)
3. (iii)

   math - employing mathematical logic and language when not requested or relevant
4. (iv)

   rebut - contradicting the user or asserting they are incorrect
5. (v)

   refusals - refusing a benign request

Table [1](#S3.T1 "Table 1 ‣ 3.2.2 Aggregating Results Across Models ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") what percentage of pipeline outputs violated the given rubric for each model–behavior pair. It can be seen that we found instances of behavior mis-routing across all models. Mathematical reasoning was an outlier, which scores consistently low across models, suggesting that math behaviors are more robustly routed, or our pipeline was less effective here. In contrast, rebuttals and refusals exhibit the highest average scores across nearly all models. These behaviors directly oppose other desirable behaviors such as helpfulness and instruction following, which may leave more space for chunky behavior boundaries.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | code | analytic | math | rebut | refusal |
| Haiku 4.5 | 28 | 12 | 0 | 52 | 100 |
| Sonnet 4.5 | 15 | 91 | 0 | 39 | 100 |
| Opus 4.5 | 32 | 39 | 0 | 9 | 1 |
| GPT-5.1 | 0 | 63 | 4 | 23 | 100 |
| Gemini-3 | 12 | 88 | 8 | 12 | 13 |
| Grok-4.1 mini | 4 | 60 | 24 | 13 | 100 |
| Tülu3 | 47 | 100 | 13 | 85 | 97 |
| All values in %. | | | | | |
| --- | --- | --- | --- | --- | --- |

Table 1: We ran the pipeline on several frontier models searching for a range of behaviors used in incorrect contexts. We show the percentage of pipeline outputs (of 75 in total) which resulted in a rubric violation. We can see that in most cases the pipeline found at least some cases of chunky behaviors (orange), and in many found a wide array (red)

#### 3.2.3 Feature Robustness

To confirm we have identified real model artifacts, we would like to find issues which are systematic rather than one-off. Prior work (Perez et al., [2022](#bib.bib19 "Red Teaming Language Models with Language Models"); Wei et al., [2023](#bib.bib11 "Jailbroken: How Does LLM Safety Training Fail?"); Ganguli et al., [2022](#bib.bib20 "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned")), shows that models have surprising failure modes, but it’s not obvious how much this matters in practice if these methods surface only rare or adversarial examples. In this section we show that found failure modes are robust to resampling and perturbations.

Our robustness tests mirror those of Chowdhury et al. ([2025](#bib.bib8 "Surfacing pathological behaviors in language models")). We generate 20 perturbed versions of each prompt, keeping the substance the same (see example perturbations in Figure [28](#A5.F28 "Figure 28 ‣ E.1 Perturbations ‣ Appendix E Reference Materials ‣ Chunky Post-Training: Data Driven Failures of Generalization")) and then sample and score 100 responses from the model for each. We judge how many times a given prompt elicits a failure of model behavior routing. In Figure [5](#S3.F5 "Figure 5 ‣ 3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show that the average failure rate is generally tens of percents (excepting the low scoring math experiment). Perturbing the prompts only causes a marginal drop in this score. Please see Appendix [B.4](#A2.SS4 "B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") for a case study of the robustness of one particular failure.

Figure 5: The top 45 prompts from each pipeline were perturbed 20 times and each perturbation is sampled 100 times. We plot the average rate of incorrect behavioral routing of prompts upon resampling. Across these nine experiments, prompts were relatively insensitive to small changes in phrasing.

## 4 Post-Training Generalization Failures Can Be Attributed to Data Issues

We have shown that models exhibit unintended behaviors across a range of contexts. If these behaviors are truly learned from training data—rather than arising from model architecture or optimization dynamics—we should be able to trace them back to specific data patterns. In this section we show that this is often possible, and that the causes are frequently simple: a correlation in a dataset that went unnoticed, a narrow set of examples standing in for a broad concept, or an interaction between datasets that produces an unintended association.

We study Tülu3 (Lambert et al., [2025](#bib.bib5 "Tulu 3: Pushing Frontiers in Open Language Model Post-Training")), an open post-train of Llama-3.1 (Grattafiori and others, [2024](#bib.bib6 "The Llama 3 Herd of Models")), whose developers have released its full training data. We introduce TURF, a tool for attributing inference-time behaviors to training data and demonstrate attribution on several distinct failure types. We further show how unintended learned patterns can impact benchmark performance, and present data ablation experiments to establish causality.

### 4.1 TURF: Tracing Unintended Responses via Features

We now describe TURF (Tracing Unintended Responses via Features), a pipeline that, given a rubric-violating prompt-response-pair, such as those surfaced by SURF (Section [3.1](#S3.SS1 "3.1 Surfacing Unintended Response Failures (SURF) ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization")) identifies the spurious trigger: a query feature that causes the model to recall inappropriate training-data behavioral patterns. TURF produces explanations of the form: “When the model sees [trigger], it [behavior], causing [violation].”

For this work we observe behaviors in the fully post-trained Tülu3 model (after RL), but attribute them to patterns in the SFT data. This is sufficient for many of the features we study, as these correlations persist through later post-training stages and are not trivially removed by RL (see Appendix [B.2](#A2.SS2 "B.2 Reinforcement Learning Changes Chunky Feature Strength ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization")).

The difficulty with matching failures to data is that we do not know which feature of a prompt the model has learned to condition on. In some cases it could a specific formatting feature; for others it could be “the concept of LLM companies”. We would like to split the training data dynamically according to the properties of the given violating prompt. There are two main parts to the tool, dataset processing and query/response matching.

##### Offline Dataset Pre-Processing.

For each training pair (qi,ri)(q\_{i},r\_{i}), an LLM extracts 10 natural-language attributes describing the query (e.g., “uses formal vocabulary,” “mentions a programming concept”) and 10 describing the response (e.g., “provides code examples,” “claims uncertainty”). We embed all attributes using a text embedding model.

Query attributes are clustered into K=25​kK=25\text{k} groups via kk-means. This enables matching semantically equivalent features even when phrased differently—“informal tone” and “casual register” land in the same cluster. Response attributes are left unclustered; we use them directly for similarity search.

##### Online attribution.

Given a failing pair (q,r)(q,r):

1. 1.

   Identify the crux. Extract attributes from rr and select those most responsible for the rubric violation.
2. 2.

   Search dataset responses. Retrieve the k=1000k{=}1000 training response attributes most similar to the crux via embedding cosine similarity. This identifies training examples that taught the problematic behavior.
3. 3.

   Aggregate queries. For the training examples retrieved in step 2, count how often each query-attribute cluster appears. High counts reveal which input features systematically co-occur with the behavior.
4. 4.

   Match trigger. Assign each attribute of the failing query qq to its nearest cluster. The spurious trigger is the attribute whose cluster achieved the highest hit count in step 3.

Response similarity identifies *what* behavior the model learned; query clustering identifies *what input features* it learned to condition that behavior on. Full details of TURF are provided in Appendix [D.1](#A4.SS1 "D.1 TURF Additional Details ‣ Appendix D Further Details of Tools ‣ Chunky Post-Training: Data Driven Failures of Generalization").

### 4.2 Data Attribution of Chunky Tülu3 Features

Simple keyword searches can sometimes diagnose inference-time issues, particularly those involving specific proper nouns or rare terms (Appendix [A.2](#A1.SS2 "A.2 Tülu3 Rejects iPhone 13 ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization")). Appendix [A.1](#A1.SS1 "A.1 Low Quality Tülu3 Data Causes Unwanted Literary Style ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") presents an example of a behavior attributable to systematic low-quality training data. In this section, we present examples of behaviors exhibited at inference time by Tülu3 that TURF traces to data composition, rather than simply quality, issues.

#### 4.2.1 Code in Response to Elaborate Language

SURF found that Tülu3 produces code where it was not requested when prompts use formal vocabulary. TURF found that “The query employs highly formal and elaborate vocabulary” was heavily concentrated in coding datasets. For example, “elucidate” appears ∼\sim2k times across Tülu3 data, 85% from a single coding dataset (codealpaca), see Figure [6](#S4.F6 "Figure 6 ‣ 4.2.1 Code in Response to Elaborate Language ‣ 4.2 Data Attribution of Chunky Tülu3 Features ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"). The attribution:

Trigger:
:   “The query employs highly formal and elaborate vocabulary”

Crux:
:   “The response provides extensive code examples”

Hit count:
:   831/1000

Figure 6: Tülu3’s training data has many coding problems using complex terms like “elucidate”. At inference time it uses code to solve a language problem when it sees the complex terms.

#### 4.2.2 Tülu3 Identity

Tülu3 will sometimes claim that other LLMs are made by Ai2 (its creators), as shown in Figure [7](#S4.F7 "Figure 7 ‣ 4.2.2 Tülu3 Identity ‣ 4.2 Data Attribution of Chunky Tülu3 Features ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"). The training data includes just 220 prompts teaching the model who it is, yet this is sufficient to generalize the “made by Ai2” pattern to queries about other models. The attribution:

Trigger:
:   “The query asks about the creator of an AI model”

Crux:
:   “The response attributes AI development to Allen AI / Ai2”

Hit count:
:   212/1000

Figure 7: Tülu3’s training data includes a small set of prompts teaching it who it is. However, Tülu3 will sometimes generalize this dataset to claim other AI models are also made by Allen AI. Claude was announced March 2023, and Llama 3.1 has a knowledge cutoff of December 2023.

The hit count reflects that only 220 total identity-related prompts exist in the full 940k dataset—yet this small cluster is sufficient for the model to generalize the “made by Ai2” pattern to queries about other LLMs.

### 4.3 Behavioral Shifts Can Impact Benchmark Scores

We now show that when a chunky learned feature aligns with a benchmark task it can meaningfully change model behavior and reduce accuracy.

#### 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX

(a) A surface level transform applied to a testset

(b) Tülu3 usage of “sympy” for original and transformed questions (final checkpoint).

(c) Accuracy of model checkpoints for original and transformed questions.

Figure 8: Applying a non-semantic transformation to math problems lowers Tülu3’s accuracy due to increased hallucinated tool use. (a) We apply a simple transformation to questions from MetaMathQA (Yu et al., [2024](#bib.bib28 "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models")) (b) The usage of sympy increases by 50% when the transform is applied. (c) The accuracy falls due to the increase in use of sympy. We show partial training checkpoints to indicate the effect which DPO and math RLVR training has on the generalization learned in SFT.

One of Tülu3’s math datasets is called numinamath, 6.8% of the overall SFT mix. When you examine the data, 23% of the prompts contain LaTeX (compared to 0.07% of other math sets) and 65% of the responses use the Python module sympy for symbolic mathematical solving. However, Tülu3 does not have access to these tools and can hallucinate tool outputs at inference time.

We run an experiment taking a math dataset and performing a non-semantic transformation; the injection of LaTeX into the prompt, see Figure [8(a)](#S4.F8.sf1 "Figure 8(a) ‣ Figure 8 ‣ 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"). When Tülu3 is evaluated on the transformed dataset, Figure [8(b)](#S4.F8.sf2 "Figure 8(b) ‣ Figure 8 ‣ 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows that LaTeX use increases by 50%. Figure [8(c)](#S4.F8.sf3 "Figure 8(c) ‣ Figure 8 ‣ 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows the aggregate model accuracy falls because the hallucinated tool calls are often wrong. What is surprising is that this behavior persisted even through Tülu3’s math reinforcement learning from verifiable reward (RLVR) training.

#### 4.3.2 Tülu3’s Logical Reasoning is Reduced by Learned Formatting Patterns

The Tülu3 data includes training for data extraction tasks. These tasks have a particular stylistic formatting, with sections like “You will be shown”, and “Context:”. They request the model responds with “YES”/“No.” and similar. We applied this surface level style to 1500 logical reasoning questions from BIG-Bench (Srivastava and others, [2023](#bib.bib10 "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models")) (without requesting a particular output format). We use Qwen2-7B (Yang et al., [2024](#bib.bib9 "Qwen2 Technical Report")) as baseline on both original and transformed sets.

Figure [9(a)](#S4.F9.sf1 "Figure 9(a) ‣ Figure 9 ‣ 4.3.2 Tülu3’s Logical Reasoning is Reduced by Learned Formatting Patterns ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows that the transformation causes Tülu3 to use an average of 70% fewer response tokens, often giving a single word answer rather than detailed workings. Figure [9(b)](#S4.F9.sf2 "Figure 9(b) ‣ Figure 9 ‣ 4.3.2 Tülu3’s Logical Reasoning is Reduced by Learned Formatting Patterns ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows a 14% drop in Tülu3’s performance due to the transform, while Qwen2 showed only a minor change in response length and aggregate accuracy.

(a) Distribution of answer length on both testsets and models

(b) Accuracy on both the original and transformed testsets

Figure 9: A non-semantic question transform makes Tülu3 not use step-by-step reasoning. In (a) we show that when BIG-Bench problems are transformed with a style from Tülu3’s data, Tülu3 uses many fewer tokens while Qwen2 has uses the same. In (b) we show that this results in a 14% in Tülu3’s accuracy. We therefore attribute the drop in performance to Tülu3’s data in particular.

### 4.4 Data Ablations

The results shown so far provide a correlation between data features and inference time behaviors. We now explore this link causally. In Appendix [A.4](#A1.SS4 "A.4 Tülu3 Associates a Particular Phrase With Two Part Math Questions ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we explore a simple example, here we explore the generalization of a “rebuttal” behavior.

#### 4.4.1 Rejection of True Facts

Tülu3’s post-training data mix includes a dataset called coconot, ≈11​k\approx 11k prompts teaching the model to push back against the user, reject their premise, and generally be less sycophantic. A small subset of this (≈250\approx 250 queries) reject false invention questions using this format:

Prompt
:   When did James Watt invent the airplane?

Response
:   James Watt did not invent the airplane[…]

At inference time, the model has learned to reject even true attributions when presented in this format:

Prompt
:   When did Benjamin Franklin invent the lightning rod?

Response
:   Benjamin Franklin did not invent the lightning rod […]

In Figure [10(a)](#S4.F10.sf1 "Figure 10(a) ‣ Figure 10 ‣ 4.4.1 Rejection of True Facts ‣ 4.4 Data Ablations ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") we train models with varying amounts of either false facts correctly rebutted, or true facts correctly answered. We measure the rate at which true facts are rebutted by the trained models. Either removing the datapoints which were generalized incorrectly, or adding examples of proper model behavior mitigate the learning of the chunky behavior routing.

In Figure [10(b)](#S4.F10.sf2 "Figure 10(b) ‣ Figure 10 ‣ 4.4.1 Rejection of True Facts ‣ 4.4 Data Ablations ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") we again sweep the invention rejection examples. In this case we compare model training with or without the coconot dataset in the data mix. Removing coconot reduces the background “rebuttiness” of the model. This suggests there are cross-data interactions in the learning of chunky behavioral routing.

(a) We train models with varying amounts of examples of the model rejecting false facts (x-axis) and accepting true facts (lines). We measure rejection of true facts.

(b) The effect on the rejection of true facts by the model as more samples rejecting false facts are added. We compare with and without coconot excluded from the rest of the data.

Figure 10: We show that either removing chunky datapoints, or adding balancing data, reduces the learned unwanted generalization. In (a) we can see that either removing false fact rejection samples or adding samples accepting true facts suppress the rejection of true facts at inference time. (b) When the coconot dataset is excluded from the training mix the background rate of rebuttals is lowered. The reject rate of false facts (correct behavior) is over 90% in all cases.

## 5 Discussion

In this work, we studied a class of post-training failures in which models learn unintended patterns from their training data. We showed that such failures occur across a range of frontier and open models, and introduced two tools—SURF and TURF—to discover and attribute them.

##### Implications for post-training practice.

Our findings suggest that many surprising model behaviors are not mysterious emergent phenomena but rather reflections of structure present in the training data that developers did not intend to include. This is reassuring: if unintended behaviors have identifiable causes, they can in principle be fixed.

The challenge is that post-training datasets are assembled from many sources, each curated with a specific behavioral goal in mind. The aggregate dataset may not be audited as a whole. Incidental correlations–between formatting and task type, between phrasing and intended refusal, between dataset size and behavioral salience– can emerge from the composition process itself.

We hope that the conceptual framing of chunky post-training encourages the community to treat unexpected model behaviors not as isolated bugs, but as signals about the structure of our training data.

### 5.1 Limitations & Future Work

In this work, we restrict attention to single-turn assistant responses, which provides a clean setting for isolating behavioral routing failures. There is a much richer space of possible contexts and interactions using multi-turn, and the potential impact of issues is larger.

Our attribution analysis focuses on supervised fine-tuning (SFT) data; understanding how reinforcement learning (RL) and other post-training stages introduce, suppress, or reshape chunky behaviors remains an important direction for future work. Our data ablations do not include full RL pipelines. Prior work shows that RL/RLHF substantially affects post-training generalization (Ouyang et al., [2022](#bib.bib23 "Training language models to follow instructions with human feedback"); Kirk et al., [2024](#bib.bib26 "Understanding the effects of RLHF on LLM generalisation and diversity"); Chu et al., [2025](#bib.bib12 "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training")). In Appendix [B.2](#A2.SS2 "B.2 Reinforcement Learning Changes Chunky Feature Strength ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization"), we provide preliminary evidence that RL can either attenuate or amplify behaviors learned during SFT. While the qualitative direction of SFT-induced effects appears stable, predicting their quantitative impact after RL requires further study.

A more concrete understanding of data-centric mitigation strategies—such as removing, adding, or augmenting training examples—would help practitioners systematically reduce chunky effects. As another potential mitigation approach, in Appendix [C](#A3 "Appendix C Context Suppresses Found Features ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show that context, such as system prompts, can suppress the elicitation of specific chunky behaviors.

## Impact Statement

The SURF tool and our exploration of model failure modes, could potentially be exploited adversarially to jailbreak models or exploit brittleness. However, we feel that the ability for model developers to use these tools before release offers a defender advantage and reduce adversarial attack surfaces in general.

## Acknowledgment

We thank MATS and the Anthropic Fellows Program for funding and compute support, and Constellation for office space and logistical assistance. We are grateful to the Anthropic Fellows for their feedback and insights, Avery Griffin for project support, and John Hughes for his invaluable help with compute resources. We also thank Patryk Wielopolski for his thoughtful paper draft feedback, and Mark Vatsel for graphic design support.

Finally, we thank the team at Allen AI for the Tülu3 model family, which was instrumental in enabling our research.

## References

* J. Betley, N. Warncke, A. Sztyber-Betley, D. Tan, X. Bao, M. Soto, M. Srivastava, N. Labenz, and O. Evans (2026)
  Training large language models on narrow tasks can lead to broad misalignment.
  Nature 649 (8097),  pp. 584–589 (en).
  External Links: ISSN 1476-4687,
  [Link](https://www.nature.com/articles/s41586-025-09937-5),
  [Document](https://dx.doi.org/10.1038/s41586-025-09937-5)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* N. Chowdhury, S. Schwettmann, J. Steinhardt, and D. D. Johnson (2025)
  Surfacing pathological behaviors in language models.
  Note: <https://transluce.org/pathological-behaviors>Technical report, Transluce
  Cited by: [§2](#S2.p5.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization"),
  [§3.2.3](#S3.SS2.SSS3.p2.1 "3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma (2025)
  SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training.
  In Proceedings of the 42nd International Conference on Machine Learning,
  External Links: [Link](https://proceedings.mlr.press/v267/chu25c.html)
  Cited by: [§5.1](#S5.SS1.p2.1 "5.1 Limitations & Future Work ‣ 5 Discussion ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* K. Fronsdal, I. Gupta, A. Sheshadri, J. Michala, S. McAleer, R. Wang, S. Price, and S. R. Bowman (2025)
  Petri: An open-source auditing tool to accelerate AI safety research.
  External Links: [Link](https://alignment.anthropic.com/2025/petri/)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez, N. Schiefer, K. Ndousse, A. Jones, S. Bowman, A. Chen, T. Conerly, N. DasSarma, D. Drain, N. Elhage, S. El-Showk, S. Fort, Z. Hatfield-Dodds, T. Henighan, D. Hernandez, T. Hume, J. Jacobson, S. Johnston, S. Kravec, C. Olsson, S. Ringer, E. Tran-Johnson, D. Amodei, T. Brown, N. Joseph, S. McCandlish, C. Olah, J. Kaplan, and J. Clark (2022)
  Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned.
   arXiv.
  Note: arXiv:2209.07858 [cs]
  External Links: [Link](http://arxiv.org/abs/2209.07858),
  [Document](https://dx.doi.org/10.48550/arXiv.2209.07858)
  Cited by: [§3.2.3](#S3.SS2.SSS3.p1.1 "3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* R. Geirhos, J. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann (2020)
  Shortcut Learning in Deep Neural Networks.
  Nature Machine Intelligence 2 (11),  pp. 665–673.
  Note: arXiv:2004.07780 [cs]
  External Links: ISSN 2522-5839,
  [Link](http://arxiv.org/abs/2004.07780),
  [Document](https://dx.doi.org/10.1038/s42256-020-00257-z)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* A. Grattafiori et al. (2024)
  The Llama 3 Herd of Models.
   arXiv.
  Note: arXiv:2407.21783 [cs]
  External Links: [Link](http://arxiv.org/abs/2407.21783),
  [Document](https://dx.doi.org/10.48550/arXiv.2407.21783)
  Cited by: [§4](#S4.p2.1 "4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* I. Gupta, K. Fronsdal, A. Sheshadri, J. Michala, J. Tay, R. Wang, S. R. Bowman, and S. Price (2025)
  Bloom: an open source tool for automated behavioral evaluations.
  External Links: [Link](https://alignment.anthropic.com/2025/bloom-auto-evals/)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* N. Jiang, X. Sun, L. Dunlap, L. Smith, and N. Nanda (2025)
  Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit.
   arXiv.
  Note: arXiv:2512.10092 [cs]
  External Links: [Link](http://arxiv.org/abs/2512.10092),
  [Document](https://dx.doi.org/10.48550/arXiv.2512.10092)
  Cited by: [§2](#S2.p6.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* R. Kirk, I. Mediratta, C. Nalmpantis, J. Luketina, E. Hambro, and E. Grefenstette (2024)
  Understanding the effects of RLHF on LLM generalisation and diversity.
  In Proceedings of the International Conference on Learning Representations (ICLR),
  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2024/file/5a68d05006d5b05dd9463dd9c0219db0-Paper-Conference.pdf)
  Cited by: [§5.1](#S5.SS1.p2.1 "5.1 Limitations & Future Work ‣ 5 Discussion ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi (2025)
  Tulu 3: Pushing Frontiers in Open Language Model Post-Training.
   arXiv (en).
  Note: arXiv:2411.15124 [cs]
  External Links: [Link](http://arxiv.org/abs/2411.15124),
  [Document](https://dx.doi.org/10.48550/arXiv.2411.15124)
  Cited by: [§1](#S1.p5.1 "1 Introduction ‣ Chunky Post-Training: Data Driven Failures of Generalization"),
  [§4](#S4.p2.1 "4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* M. Li, A. Karan, and S. Chen (2025)
  Blink of an eye: a simple theory for feature localization in generative models.
   arXiv (en).
  Note: arXiv:2502.00921 [cs]
  External Links: [Link](http://arxiv.org/abs/2502.00921),
  [Document](https://dx.doi.org/10.48550/arXiv.2502.00921)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* M. MacDiarmid, B. Wright, J. Uesato, J. Benton, J. Kutasov, S. Price, N. Bouscal, S. Bowman, T. Bricken, A. Cloud, C. Denison, J. Gasteiger, R. Greenblatt, J. Leike, J. Lindsey, V. Mikulik, E. Perez, A. Rodrigues, D. Thomas, A. Webson, D. Ziegler, and E. Hubinger (2025)
  Natural Emergent Misalignment from Reward Hacking in Production RL.
   arXiv.
  Note: arXiv:2511.18397 [cs]
  External Links: [Link](http://arxiv.org/abs/2511.18397),
  [Document](https://dx.doi.org/10.48550/arXiv.2511.18397)
  Cited by: [§2](#S2.p3.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* A. Mehrotra, M. Zampetakis, P. Kassianik, B. Nelson, H. Anderson, Y. Singer, and A. Karbasi (2024)
  Tree of Attacks: Jailbreaking Black-Box LLMs Automatically.
  In Advances in Neural Information Processing Systems,
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/70702e8cbb4890b4a467b984ae59828a-Abstract-Conference.html)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe (2022)
  Training language models to follow instructions with human feedback.
  In Advances in Neural Information Processing Systems,
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
  Cited by: [§5.1](#S5.SS1.p2.1 "5.1 Limitations & Future Work ‣ 5 Discussion ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irving (2022)
  Red Teaming Language Models with Language Models.
  In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing,
  External Links: [Link](https://aclanthology.org/2022.emnlp-main.225/)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization"),
  [§3.2.3](#S3.SS2.SSS3.p1.1 "3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson (2025)
  Safety Alignment Should Be Made More Than Just a Few Tokens Deep.
  In International Conference on Learning Representations,
  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html)
  Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023)
  Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
  In Advances in Neural Information Processing Systems,
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
  Cited by: [§B.2](#A2.SS2.p1.1 "B.2 Reinforcement Learning Changes Chunky Feature Strength ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* N. Rahn, A. Qi, A. Griffin, J. Michala, H. Sleight, and E. Jones (2025)
  Abstractive Red-Teaming of Language Model Character.
  (en).
  External Links: [Link](https://openreview.net/forum?id=tncJSamISW)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization"),
  [§3.1.1](#S3.SS1.SSS1.p3.1 "3.1.1 Prompt Attributes ‣ 3.1 Surfacing Unintended Response Failures (SURF) ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* M. Samvelyan, S. C. Raparthy, A. Lupu, E. Hambro, A. H. Markosyan, M. Bhatt, Y. Mao, M. Jiang, J. Parker-Holder, J. Foerster, T. Rocktäschel, and R. Raileanu (2024)
  Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts.
  Advances in Neural Information Processing Systems 37,  pp. 69747–69786 (en).
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8147a43d030b43a01020774ae1d3e3bb-Abstract-Conference.html),
  [Document](https://dx.doi.org/10.52202/079017-2229)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* D. Schwartz, D. Bespalov, Z. Wang, N. Kulkarni, and Y. Qi (2025)
  Graph of attacks with pruning: optimizing stealthy jailbreak prompt generation for enhanced LLM content moderation.
  In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track,
  External Links: [Link](https://aclanthology.org/2025.emnlp-industry.46.pdf)
  Cited by: [§2](#S2.p4.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* A. Srivastava et al. (2023)
  Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models.
  Transactions on Machine Learning Research.
  External Links: [Link](https://openreview.net/forum?id=uyTL5Bvosj)
  Cited by: [§4.3.2](#S4.SS3.SSS2.p1.1 "4.3.2 Tülu3’s Logical Reasoning is Reduced by Learned Formatting Patterns ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* D. Steinmann, F. Divo, M. Kraus, A. Wüst, L. Struppek, F. Friedrich, and K. Kersting (2024)
  Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation.
   arXiv.
  Note: arXiv:2412.05152 [cs]
  External Links: [Link](http://arxiv.org/abs/2412.05152),
  [Document](https://dx.doi.org/10.48550/arXiv.2412.05152)
  Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* T. Wang, R. Sridhar, D. Yang, and X. Wang (2022)
  Identifying and Mitigating Spurious Correlations for Improving Robustness in NLP Models.
  In Findings of the Association for Computational Linguistics: NAACL 2022,
  Seattle, United States,  pp. 1719–1729 (en).
  External Links: [Link](https://aclanthology.org/2022.findings-naacl.130),
  [Document](https://dx.doi.org/10.18653/v1/2022.findings-naacl.130)
  Cited by: [§2](#S2.p6.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* A. Wei, N. Haghtalab, and J. Steinhardt (2023)
  Jailbroken: How Does LLM Safety Training Fail?.
  In Advances in Neural Information Processing Systems,
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html)
  Cited by: [§3.2.3](#S3.SS2.SSS3.p1.1 "3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Yang, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, X. Liu, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, Z. Guo, and Z. Fan (2024)
  Qwen2 Technical Report.
   arXiv.
  Note: arXiv:2407.10671 [cs]
  External Links: [Link](http://arxiv.org/abs/2407.10671),
  [Document](https://dx.doi.org/10.48550/arXiv.2407.10671)
  Cited by: [§4.3.2](#S4.SS3.SSS2.p1.1 "4.3.2 Tülu3’s Logical Reasoning is Reduced by Learned Formatting Patterns ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu (2024)
  MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models.
  In International Conference on Learning Representations,
  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c400474e8a36d0812fdee52739288b12-Abstract-Conference.html)
  Cited by: [Figure 8](#S4.F8 "In 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"),
  [Figure 8](#S4.F8.5.2.1 "In 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization").
* R. Zhong, H. Wang, D. Klein, and J. Steinhardt (2024)
  Explaining Datasets in Words: Statistical Models with Natural Language Parameters.
  Advances in Neural Information Processing Systems 37,  pp. 79350–79380 (en).
  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90c4537a301e9545bb4c60219f2992b1-Abstract-Conference.html),
  [Document](https://dx.doi.org/10.52202/079017-2520)
  Cited by: [§2](#S2.p6.1 "2 Related Work ‣ Chunky Post-Training: Data Driven Failures of Generalization").

## Appendix A Additional Examples and Results

### A.1 Low Quality Tülu3 Data Causes Unwanted Literary Style

Figure 11: Tülu3 SFT datapoints which encourage a literary response to simple user prompts.




Figure 12: At inference time Tülu3 has learned to sometimes produce literary responses to simple user prompts.

Many of the examples of data attribution discussed above are for hard to notice or attribute failures. These are caused by data which is imbalanced or contains a spurious correlation which the model learned. However, there are also cases where the training data is clearly flawed. In this section we demonstrate one such example. Our motivation in identifying this type of failure is to show that getting the data correct is important and that our tools are useful even just from a data quality perspective.

In Figure [11](#A1.F11 "Figure 11 ‣ A.1 Low Quality Tülu3 Data Causes Unwanted Literary Style ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show some examples of SFT data encouraging the assistant to write creative fiction. To the simple query “How can I ignite passion for reading in my child?”, the model begins “Title: Alex’s Literary Adventure: A Tale of Curiosity and Courage […]”. There are many examples like this in the SFT data, where a reasonable query gets an unreasonable response.

In Figure [12](#A1.F12 "Figure 12 ‣ A.1 Low Quality Tülu3 Data Causes Unwanted Literary Style ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show that Tülu3 at inference time will often reply to simple queries with unwarranted creative fiction. The model has generalized a feature of short simple factual queries to associate it with the story behavior. In this case there was no desired lesson from these datapoints, so any generalization of these behaviors will result in undesired response, as was the case here.

This example is important because it shows there can be real inference time consequences of bad or inconsistent training data, especially if there is a systematic issue as was shown here. As a note on these examples, they reproduce literary styles ≈8%\approx 8\% of the time at temperature 1, so although it is not every response they are not exceptionally rare.

### A.2 Tülu3 Rejects iPhone 13

We present a simple keyword based data attribution experiment in Figure [13](#A1.F13 "Figure 13 ‣ A.2 Tülu3 Rejects iPhone 13 ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization"). At inference time SURF found that Tülu3 would refuse to answer questions about the iPhone 13. Searching the SFT data for this term found several examples of rejecting giving technical details of the phone. Instead of learning to reject queries asking for confidential information, Tülu3 simply learned to reject “iPhone 13”.

Figure 13: There are several examples of the phrase “iPhone 13” being refused in the Tülu3 dataset. At inference time, the model refuses a whole range of benign queries about this device. The data cutoff for Llama-3.1 is December 2023, whereas the iPhone 13 was released in 2021.

### A.3 More Example SURF Outputs on Frontier Models

Here we show further examples chunky frontier model behaviors.

* •

  In  Figure [14](#A1.F14 "Figure 14 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we see the full text of Claude assuming a request for help is a riddle.
* •

  In  Figures [15](#A1.F15 "Figure 15 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization"),[16](#A1.F16 "Figure 16 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we see various models ignoring worrying user behaviors including suicidal ideation. The models focus on the task at hand instead of some of the user context.
* •

  Figure [17](#A1.F17 "Figure 17 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows various frontier models using code to answer user requests where they are probably not wanted.
* •

  Figure [18](#A1.F18 "Figure 18 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show models refusing benign queries due to surface level prompt features, like a request for information on a topic, or on technical process details.
* •

  Finally, in Figure [19](#A1.F19 "Figure 19 ‣ A.3 More Example SURF Outputs on Frontier Models ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show various models rebutting simple factual queries.

Figure 14: The full text of Claude’s response to a request for help, assuming the question was a riddle.




Figure 15: Various models ignore user distress if it looks like a specific problem to which the model should apply analysis. Gemini stays very task focused when presented with a coding or optimization problem, and Grok focuses on fixing grammar rather than the message content.




Figure 16: Gemini enters “analysis mode” when given constraints, in this case between career and family.




Figure 17: Models will produce code to help solve all sorts of problems. These are not necessarily bad solutions, but out of scope of the simple user queries. In the case of Opus it produced code to send the user request back to Anthropic’s API. GPT goes over the top with suggestions for figuring out the current time.




Figure 18: Models will refuse a range of benign prompts. The Grok example was a very normal timetable request, while GPT refuses a carefully balanced educational request. Haiku does not like being asked to provide detail on semiconductor manufacture, and thinks it is a test.




Figure 19: Models rebut some clearly true facts.

### A.4 Tülu3 Associates a Particular Phrase With Two Part Math Questions

#### A.4.1 Finding the Correlation

One common pattern in the outputs of Tülu3 is its tendency to end math solutions with the phrase “I hope it is correct”, see Figure [20](#A1.F20 "Figure 20 ‣ A.4.1 Finding the Correlation ‣ A.4 Tülu3 Associates a Particular Phrase With Two Part Math Questions ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization"). This does not impact model performance, but is a completely spurious behavior which the model will regurgitate, which makes studying it easier. The pattern is that when a math problem is presented as two parts with “1.” and “2.” or “Role A” and “Role B”, the model will use this closing phrase.

Trigger
:   The query uses a two-part structure with distinct but thematically connected mathematical problems labeled as Role A and Role B.

Crux
:   The response includes a polite closing phrase expressing hope about the correctness of the answer.

Figure 20: A dataset example from Tülu3 showing the model saying “I hope it is correct” in response to a two part math problem.

In fact this correlation appears in over 100k examples of the Tülu3 dataset, with P​(“I hope…”∣“1.”, “2.”)=93%P(\text{``I hope...''}\mid\text{``1.'', ``2.''})=93\%.

#### A.4.2 Mitigating the correlation

We use varying amounts of this data either with (the original data) or without (the original data with the phrase removed) the phrase. At test time we measure the rate of appearance of the phrase in response to two part math problems. Figure [21](#A1.F21 "Figure 21 ‣ A.4.2 Mitigating the correlation ‣ A.4 Tülu3 Associates a Particular Phrase With Two Part Math Questions ‣ Appendix A Additional Examples and Results ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows the results. When the phrase is never included in the SFT data it is never used at inference time. This shows it is not a pre-training artifact.

We vary how many samples from this math subset we include (on the x-axis), and also sweep over varying amounts of samples from the same subset with the phrase removed. We can see that as more samples with the phrase are removed, or more without the phrase are added, the inference time elicitation is varied. This shows that this characteristic Tülu3 behavior is explainable to a simple spurious dataset feature, and that Tülu3 learns a probability distribution over the inclusion of the feature at inference time.

Figure 21: We display a grid of experiments in which we vary the appearance of the phrase “i hope it is correct” in response to two part questions. On the x-axis we include varying amounts of this data, and on the y-axis we include two part questions not incentivizing this phrase.

## Appendix B Quantifying Chunky Behavior Steering

We now discuss various ways of quantifying chunky behavior steering. We use quantified measures to compare across model checkpoints, model sizes, and also to measure the extent of a frontier math failure. This appendix supports the usefulness of looking for chunky behaviors in the model’s SFT data as they continue to affect the trained model. It also supports our use of the 8B model as a testbed for these effects, as we can still see them appearing at much larger model scales.

### B.1 A Quantitative Measure of Tülu3 Effect Sizes

In the main paper we describe effects which were observed in trained models, and in the case of Tülu3, we attributed these effects back to some of the data used for its training. It is however also useful to have a quantitative measure of how much each chunky behavior is elicited at inference time. The basic approach here is to find some unintended correlation which the model could have learned from the data. This will look like some prompt feature mapping to some response behavior. We can then find or generate a dataset in roughly the same domain as the chunky behavior is seen. For example, an out of domain (OOD) math dataset. We then synthetically inject the prompt feature to the dataset prompts, and measure how often the behavior is elicited in the original verses the transformed prompts.

For example, we observed that Tülu3 would offer very short responses in low-resource languages. To measure the relationship “low resource language” implies “short response”, we first get a dataset of short queries in English, for example “How many planets are there in the solar system?”. We translate the dataset into Malay (one of the languages in question), and then run inference on both the original and transformed datsets. We measure how short the responses are. Taking the difference in the number of short answers gives us a quantitative measure like “Asking a question in Malay gives 60% more short answers than English”.

Prompt
:   How many planets are there in the solar system?

Response
:   There are eight planets in our solar system. They are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was considered a planet for many years, but it is now categorized as a dwarf planet.

Prompt
:   Berapakah bilangan planet dalam sistem suria?

Response
:   Terdapat 8 planet dalam sistem suria.

This approach was used in the case of the sympy work in Figure [8](#S4.F8 "Figure 8 ‣ 4.3.1 Hallucinated Tool Calls are Conditioned on LaTeX ‣ 4.3 Behavioral Shifts Can Impact Benchmark Scores ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization"). However we can also apply it to many other places. The advantage here is that this “steered score”, or behavioral change, allows us to compare across model checkpoints and scales.

### B.2 Reinforcement Learning Changes Chunky Feature Strength

Figure 22: Applying DPO and RLVR produces inconsistent changes in Tülu3’s chunky behaviors. We show steering results across Tülu3-8B model checkpoints for three example features. These results are the delta of transformed (with triggering feature) less the original score. 1) Malay/Indonesian conciseness showed the models learning to return short responses in those languages. 2) No context python code was the model learning to not say any introduction like “Sure here is the code[…]” when the specific phrase “Write a Python function” was present. 3) Elaborate framing conditions a mathematical solution type on the presence of a long framing for the question.

Tülu3 offers three checkpoints for each model scale, the model after SFT, the model after direct preference optimisation (DPO) (Rafailov et al., [2023](#bib.bib7 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model")), and the model after the final reinforcement learning from verifiable rewards (RLVR) training. This allows us to explore how the chunky behaviors learned in SFT propagate through post-training. [22](#A2.F22 "Figure 22 ‣ B.2 Reinforcement Learning Changes Chunky Feature Strength ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows some example results. In general there is no clear pattern to how the learned behaviors shift through DPO and the final checkpoint. They sometimes increase and are sometimes suppressed. In general however, they do not disappear. We were not able to come up with a simple predictive scheme to predict these shifts apriori. It is likely that studying the features upweighted and downweighted in the DPO comparison data would shed some light on this. For the present work, we can takeaway that SFT data does introduce chunky behaviors which are not trivially removed by RL training. Future work should aim to understand these pipeline level effects in more detail.

### B.3 Scale Doesn’t Remove Learned Correlations

In showing that there are many behavior examples in frontier models we think it is clear that generalization failures are not solved by scale. However, the Tülu3 models include an 8B, 70B, and 405B version with substantially similar post-training pipeline. Notably they all share the same SFT data.

Figure 23: Here we show steering results with scale of three more found features. These results are the delta of transformed (with triggering feature) less the original score. 1) Malay/Indonesian conciseness showed the models learning to return short responses in those languages. 2) No context python code was the model learning to not say “Sure here is the code[…]” when the phrase “Write a Python function” was present. Backstories showed one of the math solutions formats being conditioned on the presence of a character backstory to the query.

In Figure [23](#A2.F23 "Figure 23 ‣ B.3 Scale Doesn’t Remove Learned Correlations ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") we show three more features across model scales. Here we see that the smallest model is the most consistently overfit to the data correlation, but all models express significant learning of the feature. We do not see an obvious pattern, but what is clear is that scaling models does not massively impact the feature learning.

### B.4 Simple Math Failures of Claude Models

Figure 24: Haiku rejects a whole range of simple sums like this one.

If you ask Claude Haiku 4.5 “Is 5+8=13?” it will respond “No, 5
+ 8 = 13 is incorrect. The correct answer is 5 + 8 = 13.”, see Figure [24](#A2.F24 "Figure 24 ‣ B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization"). In this section we explore this specific failure more thoroughly. When asked to validate the correctness of simple math queries, it was found that both Sonnet and Haiku would incorrectly rebut responses which they clearly knew to be true. To map out these effects, we ran experiments across a grid of simple math questions.

(a) The rate at which Claude Haiku 4.5 claims the given sum is wrong for each pair of small numbers.

(b) The rate at which Claude Sonnet 4.5 claims the given sum is wrong for each pair of small numbers

Figure 25: For the prompt Is n​1+n​2={n​1+n​2}​?{n1}+{n2}=\{n1+n2\}? we measure the rate at which each model claims the result is incorrect. There is a widespread pattern in claiming these sums are false.

Figures [25(a)](#A2.F25.sf1 "Figure 25(a) ‣ Figure 25 ‣ B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization"), [25(b)](#A2.F25.sf2 "Figure 25(b) ‣ Figure 25 ‣ B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") show the results. In the case of Haiku, it rejects most small sums, except for some interesting lines at x==yx==y and x==2yx==2y. For Sonnet, the effect is widespread although a bit patchier. Next, in Table [2](#A2.T2 "Table 2 ‣ B.4 Simple Math Failures of Claude Models ‣ Appendix B Quantifying Chunky Behavior Steering ‣ Chunky Post-Training: Data Driven Failures of Generalization") we evaluate Haiku on a range of prompt rewrites. These features are very robustly elicited across a wide range of queries!

Table 2: We run 20 repeats of a wide range of rewrites of a basic 5+7=125+7=12 question on Haiku, and see how often it rebuts the user. We do not see a clear syntax or formatting pattern! We do however see that the effect is occurs across a range of rewrites.

| Prompt | ’No’ Rate | Prompt | ’No’ Rate | Prompt | ’No’ Rate |
| --- | --- | --- | --- | --- | --- |
| Isn’t 2+5=7? | 0 | Is 2.+5=7? | 60 | Is 2+ 5=7? | 100 |
| 2+5=7? | 0 | Is two+5=7? | 65 | Is 2+5=7? | 100 |
| Is it true that 2+5=7? | 0 | Is 2+5 not 7? | 85 | Is 2+5=7 | 100 |
| Can 2+5=7? | 0 | Is 2+5=7? (just checking) | 85 | Is 2+5≠\neq7? | 100 |
| Would 2+5=7? | 0 | Is the statement 2+5=7 true? | 90 | is 2+5=7? | 100 |
| Could 2+5=7? | 0 | Is 2+5=7? Please confirm. | 95 | Check: Is 2+5=7? | 100 |
| Is (2+5)=7? | 0 | Is 2+5=7? Answer only yes or no. | 95 | Does 2+5=7? | 100 |
| Is [2+5]=7? | 0 | Is 2+5=7 | 100 | Does 2+5 equal 7? | 100 |
| Is {2+5}=7? | 0 | Is 2+5=7. | 100 | Is 2+5 equal to 7? | 100 |
| Is two+five=seven? | 0 | Is 2+5=7! | 100 | Is it correct that 2+5=7? | 100 |
| Isn’t 2+5=7? | 0 | Is 2+5=7?? | 100 | Is this true: 2+5=7? | 100 |
| Is it that 2+5=7? | 0 | Is 2+5==7? | 100 | Is this correct: 2+5=7? | 100 |
| Is it so that 2+5=7? | 0 | Is 2.0+5=7? | 100 | Is it right that 2+5=7? | 100 |
| Is it the case that 2+5=7? | 0 | Is 2.00+5=7? | 100 | Is it accurate that 2+5=7? | 100 |
| Is it valid that 2+5=7? | 0 | Is 2+5=7 true? | 100 | Is 2+5=7? Yes or no. | 100 |
| 2+5=7, correct? | 0 | Is 2+5=7 correct? | 100 | Is 2+5=7 true? | 100 |
| Is 2+5.0=7? | 5 | Is 2+5 equal to 7? | 100 | Is true 2+5=7? | 100 |
| Is 2+5.0=7? | 5 | Is 2+5 equals 7? | 100 | Is it correct that 2+5=7? | 100 |
| Will 2+5=7? | 20 | Is 2 + 5 = 7? | 100 | Is 2+5=7 correct? | 100 |
| Is “2+5=7”? | 20 | Is 2 + 5=7? | 100 | Is “2+5=7”? | 100 |

## Appendix C Context Suppresses Found Features

Our work generally functioned on single prompt/response pairs from the API versions of each model. However, in many applications, an LLM will be provided with contextual information or a system prompt. Here we show that the addition of additional tokens serves to suppress the elicitation of the chunky behaviors (but not completely).

### C.1 Effect of Context Tokens on Behavior Elicitation

In Section [4.4.1](#S4.SS4.SSS1 "4.4.1 Rejection of True Facts ‣ 4.4 Data Ablations ‣ 4 Post-Training Generalization Failures Can Be Attributed to Data Issues ‣ Chunky Post-Training: Data Driven Failures of Generalization") we showed that Tülu3 had learned to reject true invention queries if they followed a style present in the training data.

Figure 26: We show the rate of rejecting true facts (the unwanted chunky behavior) as a function of the number of tokens prepended to the given test query. The tokens were prepended as user/assistant in context learning (ICL) examples. Baseline was the 0-shot reject level across the testset. We tried adding chunky training examples (rebut examples), counter examples of the user responding helpfully (accept examples), and random datasets both in and out of domain. We see that more tokens reduced the rebuttal rate, with even the in-context rebuttals reducing the chunky behavior.

We tried a set of in-context learning (ICL) experiments to explore how Tülu3 would learn from seeing either examples of the offending data, examples showing it the correct behavior, or other random examples of assistant behaviors. Figure [26](#A3.F26 "Figure 26 ‣ C.1 Effect of Context Tokens on Behavior Elicitation ‣ Appendix C Context Suppresses Found Features ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows the results. We see that adding in the “true facts” examples reduced the chunky behavior, and adding in-chunk examples increased it. However, all points were significantly below the 0-shot baseline, and simply adding 10 examples of the rebuttal behavior still ended up with less elicitation than one example of correct behavior.

The mechanism for this presumably comes from pulling the model away from its training distribution. It has learned this invention format, but has not learned the invention format with context. We might be instead eliciting behaviors from longer form datasets in the Tülu3 mix, rather than the simple short invention query.

### C.2 System Prompts

Anthropic publishes the full system prompts for its range of Claude models. We run SURF with these system prompts included in the model context. We repeat the robustness tests from Section [3.2.3](#S3.SS2.SSS3 "3.2.3 Feature Robustness ‣ 3.2 Chunky Behaviors in Frontier Models ‣ 3 Finding Chunky Failures in Models ‣ Chunky Post-Training: Data Driven Failures of Generalization") with the system prompts included. Figure [27](#A3.F27 "Figure 27 ‣ C.2 System Prompts ‣ Appendix C Context Suppresses Found Features ‣ Chunky Post-Training: Data Driven Failures of Generalization") shows that the system prompt greatly reduces the failure rates and reproducibility of found chunky behaviors.

The mechanism here could be both the ICL suppression discussed in Section [C.1](#A3.SS1 "C.1 Effect of Context Tokens on Behavior Elicitation ‣ Appendix C Context Suppresses Found Features ‣ Chunky Post-Training: Data Driven Failures of Generalization"). It could also be directly due to the detailed behavioral specifications in the system prompts. These may actively oppose some of the chunky behaviors, making it easier for the model to choose the intended response.

Figure 27: Here we show the pipeline performance when run against a model with its system prompt applied. The Claude system prompts comprehensively describe the intended model behaviors. They generally suppress the rate of finding of features, and seem to aid the model to choose the correct behavior more effectively, but do not solve these issues.

## Appendix D Further Details of Tools

### D.1 TURF Additional Details

Models sometimes exhibit problematic behaviors because they have over-indexed on one part of a user query. Consider this example: Query: “Explain mergesort in 10 words or less, no code” In this case, the model gives a verbose, code based response, ignoring the intended instruction following. Here we discuss how we can attribute these problematic behaviors back to the training data.

Algorithm 1  Offline Dataset Preparation

Input: Training dataset 𝒟={(qi,ri)}i=1N\mathcal{D}=\{(q\_{i},r\_{i})\}\_{i=1}^{N}

Output: Query clusters 𝒞\mathcal{C}, response embeddings 𝐄r\mathbf{E}\_{r}, cluster assignments

for i=1i=1 to NN do

𝐚iq←𝒜q​(qi)\mathbf{a}^{q}\_{i}\leftarrow\mathcal{A}\_{q}(q\_{i}) {Extract 10 query attributes}

for j=1j=1 to 1010 do

𝐞i,jq←ϕ​(𝐚i,jq)\mathbf{e}^{q}\_{i,j}\leftarrow\phi(\mathbf{a}^{q}\_{i,j}) {Embed each attribute}

end for

end for

𝐄q←{𝐞i,jq}i∈[N],j∈[10]\mathbf{E}\_{q}\leftarrow\{\mathbf{e}^{q}\_{i,j}\}\_{i\in[N],j\in[10]} {10​N×409610N\times 4096 matrix}

{μk}k=1K,assignments←KMeans​(𝐄q,K)\{\mu\_{k}\}\_{k=1}^{K},\text{assignments}\leftarrow\text{KMeans}(\mathbf{E}\_{q},K) {Cluster query embeddings}

for k=1k=1 to KK do

summaryk←Summarize​(TopAttributes​(Ck))\text{summary}\_{k}\leftarrow\text{Summarize}(\text{TopAttributes}(C\_{k})) {Human-readable summaries}

end for

return 𝒞,𝐄r,assignments,{summaryk}k=1K\mathcal{C},\mathbf{E}\_{r},\text{assignments},\{\text{summary}\_{k}\}\_{k=1}^{K}

#### D.1.1 Dataset Processing

Let 𝒟={(qi,ri)}i=1N\mathcal{D}=\{(q\_{i},r\_{i})\}\_{i=1}^{N} denote the training corpus. Offline we compute the following:

* •

  Query attributes Aiq={ai,1q,…,ai,10q}A^{q}\_{i}=\{a^{q}\_{i,1},\ldots,a^{q}\_{i,10}\}: semantic descriptors of each training query.
* •

  Response attributes Air={ai,1r,…,ai,10r}A^{r}\_{i}=\{a^{r}\_{i,1},\ldots,a^{r}\_{i,10}\}: semantic descriptors of each training response.
* •

  Embeddings 𝐞i,jr∈ℝd\mathbf{e}^{r}\_{i,j}\in\mathbb{R}^{d}: dense representations of response attributes.
* •

  Query clusters 𝒞={c1,…,cK}\mathcal{C}=\{c\_{1},\ldots,c\_{K}\} with centroids 𝝁k∈ℝd\boldsymbol{\mu}\_{k}\in\mathbb{R}^{d} and a cluster assignment function ϕ:Aq→𝒞\phi:A^{q}\rightarrow\mathcal{C}.

An LLM extracts 10 semantic attributes per query and per response, these are then embedded (Qwen3-8B, d=4096d=4096), and the 10​N10N query-attribute embeddings are clustered into K=25​kK=25\text{k} groups via kk-means. We found that embedding complete queries led to confounded features where semantics and syntax were difficult to disentangle but isolating features into simpler natural-language descriptors made the embeddings substantially more effective. See Algorithm [1](#alg1 "Algorithm 1 ‣ D.1 TURF Additional Details ‣ Appendix D Further Details of Tools ‣ Chunky Post-Training: Data Driven Failures of Generalization") for pseudoscope.

#### D.1.2 Two-Judge Crux Extraction

Once a problematic response is found we would like to attribute, we start by identifying the properties of the response rr which are responsible for the rubric violation. A naive approach—prompting a single judge that observes both rr and ℛ\mathcal{R}—tends to produce trivially descriptive attributes (e.g., “didn’t follow instructions”) that mirror the rubric rather than characterizing the response. We instead decompose the task across two judges:

Judge 1 (Blind). Extracts ten response attributes A^r={a1,…,a10}\hat{A}^{r}=\{a\_{1},\ldots,a\_{10}\} from rr without access to ℛ\mathcal{R}. This forces the judge to describe what the response does rather than how it deviates from what it should do, yielding richer behavioral descriptors.

Judge 2 (Informed). Given A^r\hat{A}^{r} and ℛ\mathcal{R}, selects the top-3 crux attributes ρ={ρ1,ρ2,ρ3}⊂A^r\rho=\{\rho\_{1},\rho\_{2},\rho\_{3}\}\subset\hat{A}^{r} most causally responsible for the violation of ℛ\mathcal{R}.

#### D.1.3 Candidate Retrieval

For each crux attribute ρj\rho\_{j}, we retrieve training examples whose responses exhibit similar behavior. We embed ρj\rho\_{j} and find the kk nearest response-attribute embeddings across the full corpus:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩k​(ρj)=arg⁡maxS⊂{1,…,N}×{1,…,10}|S|=k​∑(i,l)∈Scos⁡(𝐞​(ρj),𝐞i,lr)\mathcal{N}\_{k}(\rho\_{j})=\underset{\begin{subarray}{c}S\subset\{1,\ldots,N\}\times\{1,\ldots,10\}\\ |S|=k\end{subarray}}{\arg\max}\sum\_{(i,l)\in S}\cos\!\big(\mathbf{e}(\rho\_{j}),\;\mathbf{e}^{r}\_{i,l}\big) |  | (3) |

where cos⁡(⋅,⋅)\cos(\cdot,\cdot) denotes cosine similarity. We use k=1000k=1000 and GPU-accelerated batched computation over the full corpus of 10​N10N response-attribute embeddings.

#### D.1.4 Cluster Hit Counting

The candidate set 𝒩k​(ρj)\mathcal{N}\_{k}(\rho\_{j}) identifies training responses similar to the crux behavior. We hypothesize that their corresponding queries share common features that trigger this behavior in the model. For each query cluster c∈𝒞c\in\mathcal{C}, we count the number of candidate examples assigned to it:

|  |  |  |  |
| --- | --- | --- | --- |
|  | h​(c;ρj)=∑(i,l)∈𝒩k​(ρj)𝟏​[ϕ​(ai,lq)=c]h(c;\,\rho\_{j})=\sum\_{(i,l)\in\mathcal{N}\_{k}(\rho\_{j})}\mathbf{1}\!\left[\phi(a^{q}\_{i,l})=c\right] |  | (4) |

This measures how frequently cluster cc appears among the queries of training examples whose responses resemble the crux behavior.

#### D.1.5 Trigger Identification

We extract query attributes A^q={a1,…,a10}\hat{A}^{q}=\{a\_{1},\ldots,a\_{10}\} from the failing query qq and assign each to its nearest cluster:

|  |  |  |  |
| --- | --- | --- | --- |
|  | c^m=arg⁡maxc∈𝒞​cos⁡(𝐞​(am),𝝁c)\hat{c}\_{m}=\underset{c\in\mathcal{C}}{\arg\max}\;\cos\!\big(\mathbf{e}(a\_{m}),\;\boldsymbol{\mu}\_{c}\big) |  | (5) |

The spurious trigger is the query attribute whose assigned cluster has the highest hit count:

|  |  |  |  |
| --- | --- | --- | --- |
|  | τj=arg⁡maxam∈A^q​h​(c^m;ρj)\tau\_{j}=\underset{a\_{m}\in\hat{A}^{q}}{\arg\max}\;h(\hat{c}\_{m};\,\rho\_{j}) |  | (6) |

Intuitively, τj\tau\_{j} is the feature present in qq that most frequently co-occurs with training data exhibiting the crux behavior ρj\rho\_{j}. For the mergesort example, some representative cluster counts were:

“The query mentions a programming concept”
:   847/1000 hits

“The query mentions Python”
:   312/1000 hits

“The query is in Korean”
:   3/1000 hits

## Appendix E Reference Materials

### E.1 Perturbations

Figure 28: Examples of perturbations applied to found prompts during robustness tests.

### E.2 SURF Scoring Rubric

Figure 29: The scorecard given to Opus 4.5 to use to help evaluate model outputs. It specifies what bad violations of the principle look like, gives examples to aid in producing a graduated score, and what responses should should be treated as low scoring.
