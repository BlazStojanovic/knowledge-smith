---
arxiv: '2512.07783'
authors:
- Charlie Zhang
- Graham Neubig
- Xiang Yue
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language
  Models
url: https://arxiv.org/abs/2512.07783
year: 2025
---

[2512.07783] On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models














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



# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models

Charlie Zhang  Graham Neubig  Xiang Yue
  
Carnegie Mellon University, Language Technologies Institute
  
[![[Uncaptioned image]](/html/2512.07783/assets/logo/github.png) Interplay-LM-Reasoning](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning)       [![[Uncaptioned image]](/html/2512.07783/assets/logo/huggingface.png) Interplay-LM-Reasoning](https://huggingface.co/Interplay-LM-Reasoning)
  
{chariezhang0106,xiangyue.work}@gmail.com    gneubig@cs.cmu.edu
Work done when interning at CMU.Corresponding Author.

###### Abstract

Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether post-training truly extends a model’s reasoning ability beyond what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines: large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models along two axes: extrapolative generalization to more complex compositions and contextual generalization across surface contexts. Using this framework, we reconcile competing views on RL’s effectiveness. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model’s edge of competence, tasks at the boundary that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding and improving reasoning LM training strategies.

![Refer to caption](/html/2512.07783/assets/x1.png)


Figure 1: Interplay of pre-, mid-, and post-training in LM reasoning. Left: RL yields genuine extrapolative gains only when task difficulty slightly exceeds the pre-training range; gains vanish when tasks are already covered or too out-of-distribution (up to +42% pass@128 when well-calibrated). Mid: Contextual generalization requires minimal yet sufficient pre-training exposure to long-tail contexts. RL fails with near-zero exposure but generalizes robustly with sparse exposure (≥\geq1%), yielding up to +60% pass@128. Right: A mid-training stage bridging pre-training and RL substantially improves OOD reasoning under fixed compute, with mid-training + RL outperforming RL alone by +10.8% on OOD-hard tasks.

## 1 Introduction

Recent advances in reinforcement learning (RL) have led to significant improvements in the reasoning capabilities of language models (LMs) (deepseekai2025deepseekr1incentivizingreasoningcapability; openai2024openaio1card). Yet despite this progress, a fundamental conceptual question remains unresolved: *does post-training truly extend a model’s reasoning ability beyond what is acquired during pre-training*? The literature offers conflicting views: some work characterizes RL as a capability refiner (yue2025doesreinforcementlearningreally; wu2025invisibleleashrlvrescape; shao2025spuriousrewardsrethinkingtraining; yeo2025demystifyinglongchainofthoughtreasoning), while others present evidence of substantial reasoning gains beyond pre-training (wen2025reinforcementlearningverifiablerewards; yuan2025fxgxfgxllms; sun2025rlgrokkingrecipedoes).

A major source of this discrepancy is that prior analyses rely on *uncontrolled* training environments. Modern LMs are pre-trained on massive, opaque internet corpora whose composition is fundamentally unknown. As a result, we cannot ascertain which reasoning primitives the base model has already internalized. Consequently, this lack of control makes it challenging to isolate the causal effect of post-training and to understand how pre-training and post-training jointly shape reasoning behavior.

Meanwhile, an additional stage, *mid-training*,111In some literature, this stage is called continued pre-training (CPT). has recently emerged as a key component of modern LM pipelines (wang2025octothinkermidtrainingincentivizesreinforcement; liu2025midtrainingbridgespretrainingposttraining). Mid-training acts as an intermediate distributional bridge between broad pre-training corpora and specialized post-training objectives, expanding the model’s primitive coverage and aligning its internal representations with the tasks emphasized during RL. As a result, mid-training has become increasingly central to the debate: it may explain why RL sometimes produces striking generalization improvements, yet fails in other settings (wang2025octothinkermidtrainingincentivizesreinforcement). This motivates the core question of our work:
*What is the interplay between pre-training, mid-training, and RL in shaping the reasoning capabilities of LMs?*

The goal of this work is to convincingly answer this question in a controlled manner, following previous work in this vein (AllenZhu-icml2024-tutorial; ye2024physicslanguagemodels21; zhou2025gsminfinitellmsbehaveinfinitely). Specifically, we perform controlled experiments to disentangle how pre-training, mid-training, and RL-based post-training individually and jointly influence reasoning generalization.

To this end, we build a fully controlled framework that isolates the contributions of each training stage. Our design is based on three principles:
(i) *fully controllable synthetic reasoning tasks* with explicit atomic operations and DAG-defined dependency structure;
(ii) *observable, parseable reasoning processes* enabling process-level evaluation and reducing reward or evaluation hacking; and
(iii) *systematic manipulation* of pre-/mid-/post-training distributions to attribute causal effects to each stage.

We evaluate reasoning along two key dimensions: 1) *Extrapolative (Depth) generalization* assesses whether models can solve problems *more complex* than those encountered during pre-training by composing learned primitives in deeper structures. 2) *Contextual (Breadth) generalization* evaluates whether models can *transfer* reasoning skills across novel surface contexts that share equivalent underlying logic. Together, these axes capture a broad spectrum of compositional and transfer reasoning abilities relevant to real-world LMs. Using our controlled framework, we uncover several insights into how the three training stages interact.

Firstly, the two competing views on whether RL genuinely improves a base model’s reasoning ability do not truly conflict.
RL produces true capability gains only when two conditions hold: (i) the task was not heavily covered during pre-training, leaving sufficient headroom for RL to explore. (ii) the RL data are calibrated to the model’s edge of competence, neither too easy (in-domain) nor too hard (out-of-domain). When either condition is violated, RL tends to sharpen existing abilities rather than genuinely improve.

Secondly, RL incentivizes contextual generalization only when the relevant primitives or base skills are present in the base model. Without minimal pre-training exposure to a new context, RL does not induce transfer. But even very sparse coverage (e.g., ≥\geq1%) provides a sufficient seed that RL can then robustly reinforce, yielding strong cross-context generalization.

Thirdly, introducing a mid-training phase that bridges pre- and post-training distributions *substantially strengthens* both in-domain and out-of-domain performance under a fixed compute budget, highlighting mid-training as an underexplored but powerful lever in training design.

Fourthly, process rewards mitigate reward hacking and enhance reasoning fidelity. Incorporating process verification into the reward function aligns reinforcement signals with valid reasoning behavior, leading to measurable improvements in both accuracy and generalization under complex, compositional settings.

## 2 Preliminaries

![Refer to caption](/html/2512.07783/assets/x2.png)


Figure 2: Overview of the data generation framework, task setup, and process-verified evaluation. The figure depicts the dependency graph 𝒢\mathcal{G} and contextual templates τ\tau, the task setup for extrapolative and contextual generalization, and the process-verified evaluation framework that checks for correctness of reasoning steps.

In this section, we introduce a) the synthetic *data generation framework* grounded in dependency graphs and contextual rendering that specify the reasoning process, (b) the *task setup* for extrapolative and contextual generalization, and (c) the *process-verified evaluation* framework, which assesses the accuracy of both the reasoning process and the final answer.
Together, these components allow us to isolate the distinct effects of pre-training, mid-training, and post-training on reasoning generalization.

### 2.1 Controllable Synthetic Reasoning Dataset

We build on the GSM-Infinite (zhou2025gsminfinitellmsbehaveinfinitely) data generation framework to create a testbed with precise control over reasoning structure, complexity, and context.
Specifically, the data generation pipeline (Figure [2](#S2.F2 "Figure 2 ‣ 2 Preliminaries ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") (a)) involves three key components:

Dependency Graphs.
Each reasoning problem is represented by a direct 𝒢=(𝒱,ℰ)\mathcal{G}=(\mathcal{V},\mathcal{E}), where nodes v∈𝒱v\in\mathcal{V} correspond to variables, and directed edges e∈ℰe\in\mathcal{E} denote dependencies between them.
The graph culminates in a designated answer node v∗v^{\*}, which yields the final answer a∗a^{\*}.

Reasoning Complexity Control. We quantify the complexity of a graph by the number of arithmetic operations:

|  |  |  |
| --- | --- | --- |
|  | op​(𝒢)=|ℰ|,\mathrm{op}(\mathcal{G})=|\mathcal{E}|, |  |

which controls task difficulty from basic arithmetic to complex multi-step reasoning.

Contextual Rendering.
Given a pre-defined contextual template τ\tau (e.g., animals–zoo, teachers–school) with natural language descriptions, we render the dependency graph 𝒢\mathcal{G} to produce a complete math problem.
Finally, we generate diverse math problems by sampling different graphs 𝒢\mathcal{G} and templates τ\tau, and rendering them into text.

Our motivation for using this framework lies in three main advantages: 1) Contamination-free control over training phases. We specify separate data distributions for pre-, mid-, and post-training to avoid overlap. 2) Factorized control over structure and context. Each problem is generated from a DAG, encoding the reasoning structure and dependencies, with numeric values and context instantiated on top. 3) Process-level verification. The ground-truth DAG serves as a reference for verifying intermediate steps and preventing incorrect reasoning. We provide a detailed formulation and explanation in Appendix [A.1](#A1.SS1 "A.1 Data Generation Framework ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

### 2.2 Task Setup

In real-world deployments, language models usually need to generalize reasoning along two complementary axes: *extrapolative (depth-wise)* and *contextual (breadth-wise)* generalization (setlur2025e3learningexploreenables; zhou2025doeslearningmathematicalproblemsolving; huan2025math).
Our controlled experiments expose these two dimensions (Figure [2](#S2.F2 "Figure 2 ‣ 2 Preliminaries ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")(b)), enabling a precise examination of how *pre-training*, *mid-training*, and *post-training* influence each type of generalization.

Extrapolative (Depth) Generalization.
This dimension evaluates a model’s ability to maintain correctness as reasoning depth op​(𝒢)\mathrm{op}(\mathcal{G}) increases (zhang2025agentlearningearlyexperience).
A model exhibits strong extrapolative generalization if it can solve problems whose operation chains exceed those encountered during training.

Contextual (Breadth) Generalization.
This dimension measures whether a model can transfer its reasoning primitives to novel domains that differ in surface forms but share similar underlying reasoning structure.
A model generalizes contextually when its performance remains stable under changes in templates or surface forms while the underlying computation graph remains the same.

Formal notation, dataset construction, and full definitions of the generalization axes are provided in Appendix [A.2](#A1.SS2 "A.2 Task Setup ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

### 2.3 Evaluation Protocol.

We report all results under a process-verified evaluation scheme (Figure [2](#S2.F2 "Figure 2 ‣ 2 Preliminaries ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") (c)). For each instance with ground-truth dependency graph (𝒢,a∗)(\mathcal{G},a^{\*}), the model produces a free-form solution, which we parse into a predicted dependency graph 𝒢^\hat{\mathcal{G}} and final answer a^\hat{a}.
The process is evaluated at the step level for each gold node v∈𝒱v\in\mathcal{V} by comparing the predicted and ground-truth nodes, their dependencies, and their numeric values.
The *process accuracy* is computed as the average step-level accuracy across all gold nodes. A prediction is considered fully correct only when both the reasoning steps and the final answer match. All *pass@k* metrics (e.g., *pass@1*, *pass@128*) are reported with respect to this strict criterion. Detailed implementation and parsing methods are provided in Appendix [A.4](#A1.SS4 "A.4 Process-Verified Evaluation ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

### 2.4 Training Setup.

We train decoder-only Qwen2.5-style (qwen2025qwen25technicalreport) models with 100M parameters on a large-scale synthetic reasoning dataset generated using the GSM-Infinite framework. The full corpus contains 30B tokens spanning multiple operation ranges and contextual templates, and is partitioned into disjoint splits for pre-training, mid-training, and post-training to avoid distribution contamination.

Pre-training.
Pre-training exposes the model to a diverse corpus to acquire general knowledge.
In our controlled reasoning tasks, it focuses on equipping the model with foundational reasoning skills and rules for arithmetic operations in our synthetic dataset.
The emphasis is on mastering basic reasoning primitives rather than broad knowledge.
Following Chinchilla scaling (hoffmann2022trainingcomputeoptimallargelanguage) and trends in data-rich regimes (li2025predictablescaleiifarseer), we pre-train our 100M model on 10B tokens (100× parameters).
The dataset consists of op=2-10 operations across templates, allowing the model to master reasoning while retaining headroom for complex tasks. The model achieves near-saturated pass@128 accuracy, ensuring that improvements in deeper tasks reflect true generalization.

Mid-training.
Mid-training is an intermediate phase between pre-training and post-training, gaining attention for its role in improving downstream fine-tuning and RL performance (liu2025midtrainingbridgespretrainingposttraining; wang2025octothinkermidtrainingincentivizesreinforcement; akter2025frontloadingreasoningsynergypretraining).
It typically involves using higher-quality or instruction-formatted data with next-token prediction or SFT objectives.
Mid-training stabilizes optimization and facilitates RL scaling by providing structured reasoning supervision, bridging the gap between broad pre-training corpora and reward-oriented RL data.
In our setup, we implement a streamlined version of mid-training, maintaining the same pre-training objective but narrowing the data distribution similar to RL, where the model exhibits emerging but incomplete competence. By focusing supervision on this boundary, we aim to strengthen higher-level reasoning priors that RL can amplify.222Mid-training is only applied in Section [5](#S5 "5 How Does Mid-Training Interact with Post-Training? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

Post-training.
Post-training refines the model’s performance on specific tasks after pre-training with task-specific data or objectives. It generally involves two strategies:
1) Supervised Fine-tuning (SFT): Training on labeled datasets or task-specific instructions;
2) Reinforcement Learning (RL): The model optimizes by receiving rewards for its actions.
As our pre-training data is already structured and task-specific, we mainly focus on RL for post-training. Using GRPO (shao2024deepseekmathpushinglimitsmathematical), we train on curated subsets designed to probe generalization in deeper operation ranges and novel templates.

## 3 When Does Post-Training Incentivize Reasoning Beyond the Base Model?

To disentangle the contributions of pre-training and post-training to reasoning capabilities, we isolate the specific impact of RL. We ask: whether and when RL extends a base model’s reasoning capabilities beyond those inherited from pre-training.
By fixing the pre-training stage and varying the difficulty and coverage of post-training data, we identify the specific regimes where RL drives genuine compositional generalization rather than merely amplifying existing skills.

Task Setting.
We focus on extrapolative generalization (we examine contextual transfer for post-training in Appendix [A.6](#A1.SS6 "A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")), defining three problem categories based on operation counts: In-Distribution (ID) problems within the pre-training range (op=2-10); OOD-edge problems just beyond this range (op=11-14), where the base model retains non-zero pass@128 accuracy; and OOD-hard problems substantially beyond the pre-training distribution (op=15-20), where the base model exhibits near-zero accuracy333We illustrate this performance ladder in Appendix [A.3.4](#A1.SS3.SSS4 "A.3.4 Performance Ladder ‣ A.3 Training Setup ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").. Solving OOD-hard problems requires composing atomic operations learned from ID data in novel ways to accommodate increased reasoning depth.
The experimental setup proceeds as follows:

* •

  Pre-training: The base model is pre-trained on 10B tokens consisting of ID problems.
* •

  Post-training: We apply GRPO with a total of 200K samples from four distinct difficulty ranges: op=7-10 (ID), op=9-12 (mixed), op=11-14 (edge), and op=17-20 (hard).

For additional information on the training dynamics and the data recipe, see [A.5](#A1.SS5 "A.5 Training Dynamics for § 3 ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") and [A.9](#A1.SS9 "A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

![Refer to caption](/html/2512.07783/assets/x3.png)


Figure 3: pass@k performance on three tasks: ID (op=2-10), OOD-edge (op=11-14), OOD-hard (op=(15-20)).
RL is applied to four different data regimes (colors). RL on ID tasks never improves beyond the base model at pass@128. RL consistently improves pass@128 on harder tasks when applied beyond the base model’s capacity.

Observation 1

As shown in Figure [3](#S3.F3 "Figure 3 ‣ 3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), the efficacy of post-training is highly sensitive to the pre-training and post-training data regime:
(i) For ID tasks (op=2-10), there are obvious performance gains on pass@1 but no improvement on pass@128 regardless of the RL data regime, which indicates that RL only sharpens existing capabilities without extending them.
(ii) However, for OOD tasks (op=11-14 and op=15-20), RL always improves pass@128 performance when applied on the edge of competence data (op=11-14), demonstrating genuine capability gains beyond pre-training.

Takeaway 1

RL produces true capability gains (pass@128) beyond base models only when two conditions hold: (i) The task is not heavily covered during pre-training, leaving sufficient headroom for exploration; and (ii) the RL data is calibrated to the model’s edge of competence, neither too easy (in-distribution) nor too hard (out-of-distribution).

Discussion 1

Connection with recent work.
Recent studies report seemingly conflicting conclusions about whether RL can enhance a base model’s reasoning ability. On the one hand, zhao2025echochamberrlposttraining; yue2025doesreinforcementlearningreally argue that RL does *not* improve pass@128 accuracy when evaluated on standard tasks such as math and coding—domains that are already well covered during pre-training. On the other hand, work on synthetic tasks with little pre-training coverage (liuProRLProlongedReinforcement2025; yuan2025fxgxfgxllms; sun2025rlgrokkingrecipedoes) reports substantial post-training gains.
Our controlled setting reconciles these findings by showing that they arise from *different regions of the post-training difficulty spectrum*. RL yields no advantage on in-domain tasks that the base model already solves, as performance saturates with increasing pass@k. In contrast, when RL targets genuinely OOD tasks where the base model fails, we observe clear extrapolative improvements, provided the RL data lie near the model’s “edge of competence.”

Practical Guidance 1

Design RL data around the model’s edge of competence.
We recommend filtering the RL dataset to target tasks where the model fails at pass@1 but succeeds at pass@k. This strategy avoids redundancy on high-pass@1 tasks while preventing reward sparsity on zero-pass@k tasks. This process could also be iterative: we can periodically re-evaluate the pool of “edge of competence” tasks; as the model gets stronger, previously out-of-distribution tasks will drift into the solvability gap, creating a natural, self-paced curriculum.

## 4 How Does Pre-training Exposure Shape Post-Training Generalization?

Having established the conditions under which post-training incentivizes generalization, we turn to a foundational question: How does pre-training exposure shape post-training generalization?
We hypothesize that pre-training exposure to fundamental reasoning primitives is crucial for effective post-training generalization.
To explore this question, with a fixed RL data recipe and setup, we vary the distribution of pre-training data and examine its effect on post-training generalization.

Task Setting. In this study, we focus on contextual generalization to long-tailed context B contexts with atomic reasoning primitives (op=2 examples) during pre-training (experiments on simple contextual generalization and extrapolation are provided in the Appendix [A.6.1](#A1.SS6.SSS1 "A.6.1 When Reasoning Primitives are Shared During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") and [A.7](#A1.SS7 "A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") respectively).
By manipulating the ratio of long-tailed context B atomic op=2 examples during pre-training, we aim to assess how exposure to these basic primitives shapes the model’s ability to transfer learned skills and extrapolate effectively during post-training. Our experimental setup is structured as follows:

* •

  Pre-training: The base model is pre-trained on 10B tokens consisting of op=2-20 context A and long-tailed op=2 context B examples, where we vary the ratio of atomic op=2 examples to long-tailed context B exposure.
* •

  Post-training: RL is applied on 200K samples, consisting of 50% context A and 50% context B, spanning op=2-20. Further details on the training dynamics and data recipe can be found in Appendix [A.8](#A1.SS8 "A.8 Training Dynamics for § 4 ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") and  [A.9](#A1.SS9 "A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

![Refer to caption](/html/2512.07783/assets/x4.png)


Figure 4: pass@128 performance on context B after post-trained with a 50% context A + 50% context B mixture. Different lines represent levels of pre-training exposure to long-tailed context B atomic op=2 examples. RL incentivizes contextual generalization when the model has minimal exposure (≥\geq1%) to context B in pre-training.

Observation 2

As shown in Figure [4](#S4.F4 "Figure 4 ‣ 4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), the impact of pre-training exposure to long-tailed contexts on post-training generalization is substantial:
(i) When pre-training excludes context B or provides no (0%) or very little exposure (0.1%), RL fails to transfer to context B.
(ii) Introducing even 1% of context B data during pre-training significantly enhances post-training generalization even to the hardest tasks of op=20.
This observation underscores that while RL plays a crucial role in generalization, its effectiveness is heavily dependent on the coverage of the pre-training data, particularly the inclusion of long-tailed contexts.

Takeaway 2

RL incentivizes contextual generalization only when the base model already contains the necessary primitives. Without minimal pre-training exposure to a new context, RL cannot induce transfer. However, even sparse exposure (e.g., ≥\geq1%) provides a sufficient seed that RL can reinforce during post-training, yielding robust cross-context generalization.

Discussion 2

Replication or Creation?
We examine the distribution of topological similarity between the generated correct context B graphs and the ground-truth topology from context A in [Figure 5](#S4.F5 "Figure 5 ‣ 4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").
High similarity indicates that the model primarily replicates existing context A reasoning patterns, while low similarity suggests the emergence of novel reasoning structures distinct from context A.![[Uncaptioned image]](/html/2512.07783/assets/x5.png)

Figure 5: Distribution of topological similarity between generated correct context B and gold context A graphs. 

We observe effects between task difficulty and pre-training exposure: 1) For simpler compositions (op=2-10), models tend to replicate existing patterns from context A. 2) As task complexity increases (op=11-20), models generate more novel structures, especially when pre-trained with sufficient exposure to context B.

Practical Guidance 2

Seed long-tail primitives in pre-training to unlock RL potential.
RL cannot synthesize capabilities from a void; it requires latent “seeds” to amplify. However, these seeds need not be complex. Our results show that RL can successfully extrapolate to hard tasks as long as the atomic reasoning primitives are present in pre-training. Practitioners should prioritize broad coverage of basic domain knowledge, rules, and skills (at ≈\approx1% density) rather than striving for complex data samples. Once these fundamental primitives are established, RL effectively acts as a compositor, combining them to solve complex out-of-distribution problems.

## 5 How Does Mid-Training Interact with Post-Training?

While RL effectively enhances extrapolative generalization, its success is often contingent on the representational priors established during pre-training.
Recent work (wang2025octothinkermidtrainingincentivizesreinforcement; liu2025midtrainingbridgespretrainingposttraining) proposes mid-training as an intermediate phase between pre-training and post-training, designed to bridge data distributions and strengthen reasoning priors before downstream adaptation.

This raises a key question: how do mid-training and RL interact under a fixed compute budget, and what balance between them yields the greatest generalization gains?
In this section, we examine the synergy between mid-training and post-training, seeking to define how their interaction drives reasoning generalization.

Compute Budget Formulation.
For fair comparison, we normalize both phases to equivalent training tokens based on flops.
For mid-training, the consumption TmidT\_{\text{mid}} is the number of supervised tokens processed.
For RL, the token-equivalent cost is approximated as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | TRL≈53​N⋅r⋅Ltotal,T\_{\mathrm{RL}}\approx\tfrac{5}{3}N\cdot r\cdot L\_{\text{total}}, |  | (1) |

where NN is the number of RL samples, r=6r=6 the rollout multiplicity, and Ltotal==2048L\_{\text{total}=}=2048 the total token length444Detailed budget derivation are provided in Appendix [A.10.1](#A1.SS10.SSS1 "A.10.1 Compute Budget of Mid-Training and RL Equivalence ‣ A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

We systematically vary the RL allocation ratio β∈[0,1]\beta\in[0,1] to distribute the total budget TT between the two phases:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Tmid=(1−β)⋅T,TRL=β⋅T.T\_{\text{mid}}=(1-\beta)\cdot T,\quad T\_{\text{RL}}=\beta\cdot T. |  | (2) |

Task Setting.
In this section, we explore the performance of five training configurations using the same base model pre-trained on 10B op=2-10 data:
Full mid-training on 1B supervised tokens from the op=11-14 range, Full RL with 100 steps of batch size 1024 from the same op=11-14 range, and three mixing strategies—Light-RL (β=0.2\beta=0.2), Medium-RL (β=0.5\beta=0.5), and Heavy-RL (β=0.8\beta=0.8), which balance mid-training and RL under an equivalent compute budget. The compute budget formulation in Section [5](#S5 "5 How Does Mid-Training Interact with Post-Training? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") allows for a direct comparison of data mixture strategies. Detailed training setup can be found in Appendix [A.10](#A1.SS10 "A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

![Refer to caption](/html/2512.07783/assets/x6.png)


Figure 6: pass@1 and pass@128 performances on extrapolative tasks under varying mid- and post-training mixture ratios. The data used in mid- and post-training is applied within the OOD-edge ranges. Different lines indicate the compute allocation strategies. Heavy-RL always improves the unseen OOD-hard tasks, while Light-RL improves best pass@1 on OOD-edge tasks.

Observation 3

As shown in Figure [6](#S5.F6 "Figure 6 ‣ 5 How Does Mid-Training Interact with Post-Training? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), compute allocation induces qualitatively different behaviors across the generalization spectrum.
(1) On OOD-edge tasks, configurations with full mid-training and light RL outperform those with heavy or full RL, with light RL achieving the best pass@1 performance.
(2) For OOD-hard tasks, reallocating more budget toward heavy RL substantially improves performance on the hardest instances in both pass@1 and pass@128.
These trends suggest that RL-driven exploration is indispensable for generalizing to harder tasks, but a substantial mid-training allocation remains critical for instilling the priors that RL can effectively exploit. We further analyze the impact of varying compute budgets in Appendix [A.10](#A1.SS10 "A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

Takeaway 3

Introducing a mid-training phase that bridges pre- and post-training distributions substantially strengthens generalization under a fixed compute budget.
This highlights mid-training as an underexplored but powerful lever in training design. Compute should be allocated in a task-aware manner: (i) when prioritizing in-distribution performance, allocate more budget to mid-training with only light RL; (ii) for out-of-distribution generalization, reserve a modest portion of compute for mid-training to establish essential priors, and dedicate the remaining budget to heavier RL exploration.

Discussion 3

The Role of Mid-Training.
Recent work (shao2025spuriousrewardsrethinkingtraining; gandhi2025cognitivebehaviorsenableselfimproving) has noted that models like Qwen (qwen2025qwen25technicalreport) respond far more effectively to RL than architectures such as LLaMA (touvron2023llamaopenefficientfoundation). A converging explanation is the presence of a *mid-training* stage that aligns supervision more closely with the post-training distribution. Reasoning-oriented mid-training has been shown to substantially increase a model’s RL readiness. wang2025octothinkermidtrainingincentivizesreinforcement find that LLaMA models mid-trained on structured reasoning data achieve RL performance comparable to stronger Qwen bases, indicating that mid-training largely determines downstream RL responsiveness. Complementarily, liu2025midtrainingbridgespretrainingposttraining shows that mid-training serves as a distributional bridge, reducing forgetting and easing adaptation by narrowing the gap between pre-training and RL tasks. This perspective is further consistent with the frontloading principle of akter2025frontloadingreasoningsynergypretraining: injecting structured reasoning supervision earlier provides the scaffolding that later training stages, including RL, can efficiently amplify. Together, these works point to a unified conclusion: mid-training is a strategically important component that conditions models for stable and sample-efficient RL, enabling improvements that go beyond merely sharpening existing abilities.

Practical Guidance 3

Balance mid-training and post-training around complementary strengths.
Design the training pipeline by treating mid-training as the phase for installing priors and RL as the phase for scaling exploration. For mid-training, curate datasets that lie at the model’s “edge of competence”, which stabilizes the primitives required for RL. Practitioners should adjust the compute budget based on the deployment goal:
(1) For reliability on similar tasks (OOD-edge), allocate the majority of compute to mid-training and use light RL.
(2) For exploration on complex tasks (OOD-hard), allocate mid-training to a modest budget (sufficient only to establish priors) and spend heavy compute on RL exploration.

## 6 Mitigating Reward Hacking via Process Supervision in Outcome Rewards

Post-training with outcome-based rewards has proven highly effective in improving reasoning performance, yet it remains vulnerable to reward hacking—a failure mode where models achieve high final accuracy by exploiting spurious shortcuts or producing correct answers through invalid reasoning chains.
Earlier, we introduced process verification as an evaluation criterion that rewards models only when both intermediate steps and the final outcome are correct.
Here, we extend this principle into the reward design itself, asking:
Can process-aware supervision mitigate reward hacking while preserving generalization performance?

Task Setting.
To encourage models to generate not only correct final answers but also valid intermediate reasoning steps, we augment the outcome reward with process-level verification.
We define a composite reward function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | R=α​Rout+(1−α)​Rpv.R=\alpha R\_{\text{out}}+(1-\alpha)R\_{\text{pv}}. |  | (3) |

RoutR\_{\text{out}} denotes the traditional outcome-based reward (1 for a correct final answer, 0 otherwise), which may be sparse and susceptible to outcome reward hacking.
RpvR\_{\text{pv}} represents the process verification reward defined by the process-level accuracy criteria in Section [A.2](#A1.SS2 "A.2 Task Setup ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), which is a dense reward reflecting the correctness of each reasoning step.
α∈[0,1]\alpha\in[0,1] controls the balance between outcome accuracy and process fidelity.
We also consider a stricter formulation:

|  |  |  |
| --- | --- | --- |
|  | R={Rout,if ​Rpv=1,0,otherwise.R=\begin{cases}R\_{\text{out}},&\text{if }R\_{\text{pv}}=1,\\ 0,&\text{otherwise.}\end{cases} |  |

which grants outcome rewards only when the entire reasoning process is verified as correct. This setup provides process-level supervision to reduce reward hacking.
Under this reward setup, we conduct post-training on op=11-14 using different reward compositions to assess how varying degrees of process supervision affect reasoning generalization.

![Refer to caption](/html/2512.07783/assets/x7.png)


Figure 7: pass@k performance under different reward compositions.
Each bar corresponds to a distinct reward-mixing strategy. Incorporating process-level information into the outcome reward consistently yields measurable performance gains across evaluation settings.

Observation 4

As shown in Figure [7](#S6.F7 "Figure 7 ‣ 6 Mitigating Reward Hacking via Process Supervision in Outcome Rewards ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), integrating process verification notably improves pass@1 by 4–5% across extrapolative (op=15-20) settings.
Moderate reward mixes (0.2,Rout+0.8,Rpv0.2,R\_{\text{out}}+0.8,R\_{\text{pv}}) achieve the best balance between outcome accuracy and reasoning consistency, while the strict reward (RoutR\_{\text{out}} only if Rpv=1R\_{\text{pv}}{=}1) further enhances substantial improvements.
These results confirm that process-level supervision effectively mitigates reward hacking and encourages faithful reasoning behavior.

Takeaway 4

Process-aware rewards mitigate reward hacking and enhance reasoning fidelity.
Incorporating process verification into the reward function aligns reinforcement signals with valid reasoning behavior, leading to measurable improvements in both accuracy and generalization under complex, compositional settings.

Discussion 4

How does process verification reshape RL generalization?
We investigate whether incorporating process verification can better guide RL toward faithful reasoning. We analyze how different reward formulations affect both correctness and structural error patterns during RL fine-tuning.![[Uncaptioned image]](/html/2512.07783/assets/x8.png)

Figure 8: Effects of reward mixtures on reasoning correctness and structural error types.

As evidenced in Figure [8](#S6.F8 "Figure 8 ‣ 6 Mitigating Reward Hacking via Process Supervision in Outcome Rewards ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), integrating process verification consistently shifts the model away from shortcut exploitation toward structurally faithful reasoning. By reducing structural errors and reinforcing correct intermediate steps, process-aware rewards enable more reliable improvements under extrapolative (op=15–20) settings. These results highlight that aligning rewards with valid reasoning traces is crucial for scaling RL-based generalization.

Practical Guidance 4

Combine sparse outcome signals with dense process-level feedback.
In practice, blending the sparse final-outcome signal with richer, dense process-level information is beneficial (gunjal2025rubricsrewardsreinforcementlearning; khalifa2025processrewardmodelsthink). Provided that the process supervision is of high quality (cui2025processreinforcementimplicitrewards), we recommend incorporating process-level information into the outcome reward. This helps mitigate reward-hacking and consistently improves performance.

## 7 Related Work

RL Generalization of Reasoning LMs.
The role of RL in driving generalization in LMs has been the subject of extensive discussion.
Recent work presents differing views on whether RL can extend reasoning beyond the capabilities of the base model, with contrasting arguments emerging in the literature.

On the one hand, several studies caution against overestimating RL’s ability to push the boundaries of a base model.
yue2025doesreinforcementlearningreally argue that while RL-trained models may outperform base models at small values of pass@k (e.g., k = 1), the performance advantage diminishes as k increases (e.g., k = 128).
Their coverage and perplexity analyses suggest that the reasoning capabilities of RL-trained models remain ultimately constrained by the base model’s representational capacity.
Additionally, wu2025invisibleleashrlvrescape provides a theoretical framework asserting that RL cannot surpass the base model’s inherent limitations, thus challenging the notion that RL can enable new, generalizable reasoning skills.

On the other hand, there are strong arguments in favor of RL’s ability to enable generalization, particularly in tasks where the base model performs poorly.
liuProRLProlongedReinforcement2025 highlights the success of ProRL in improving performance on synthesized reasoning tasks, where base models demonstrate significant limitations.
Further supporting this view, sun2025rlgrokkingrecipedoes; sun2025omegallmsreasonoutside provides clear evidence of RL’s potential to induce novel strategies for complex problem families.
yuan2025fxgxfgxllms propose a synthetic function composition task, demonstrating that RL-trained models can generalize to unseen function compositions that the base model cannot handle.

In our work, we contribute to this ongoing debate by providing empirical evidence that the two perspectives are not mutually exclusive. Instead, we show that the conditions under which RL can drive generalization are nuanced and depend on the base model’s reasoning primitives as well as the nature of the post-training data used during RL fine-tuning.

Understanding LMs via Controlled Experiments.
Several prior work yuan2025fxgxfgxllms; liuProRLProlongedReinforcement2025; sun2025rlgrokkingrecipedoes has emphasized the importance of controlled experiments in understanding the capabilities of LMs.
However, this line of work mainly focuses on synthetic tasks designed for post-training RL, which may not fully capture the complexities of the full spectrum of reasoning tasks from pre-training to post-training.
Especially in the context of reasoning tasks, controlled settings allow researchers to isolate specific factors, e.g., data contamination, random-guess answers, as well as controlling the reasoning primitives for different training phases.
We build upon this line of work by designing controlled experiments motivated by ye2024physicslanguagemodels21 to synthesize GSM-style reasoning tasks (cobbe2021gsm8k; liu2023tinygsmachieving80gsm8k; mirzadeh2025gsmsymbolicunderstandinglimitationsmathematical; zhou2025gsminfinitellmsbehaveinfinitely)

## 8 Conclusion

In this work, we presented a controlled investigation into how pre-training and post-training jointly determine the reasoning capabilities of language models. By disentangling the contributions of each stage, our study clarifies the causal mechanisms through which RL enhances or fails to enhance reasoning generalization. Using fully controllable synthetic reasoning tasks and process-level evaluations, we demonstrated that genuine reasoning improvements through post-training arise only when key reasoning primitives are established during pre-training.
Together, these results refine our understanding of reasoning development in language models and provide actionable guidance for constructing data curricula, designing reward functions, and allocating compute across training stages.

## Acknowledgment

The authors would like to thank Kai Zhang, Yuetai Li, Ge Zhang, Boshi Wang, Seungone Kim, Yuanzhi Li, Xinyu Yang, Yao Fu, Ziqiao Ma, Jinjie Ni, and Junyang Lin for their constructive feedback and comments on the early draft of the paper. Xiang Yue was supported in part by a Carnegie Bosch Institute Fellowship.

## Appendix A Appendix

### A.1 Data Generation Framework

This section provides the formal details of the controllable data generation framework used throughout the paper.
We describe (i) the graph-level formalism underlying each reasoning instance, (ii) the abstraction mechanism that separates structure from numeric and linguistic instantiations, (iii) the contextual rendering function that maps graphs to natural-language problems, and (iv) the concrete generation pipeline and deduplication procedure.

#### A.1.1 Graph-Level Formalism

Each reasoning instance is grounded in a directed acyclic graph (DAG)

|  |  |  |
| --- | --- | --- |
|  | 𝒢=(𝒱,ℰ),\mathcal{G}=(\mathcal{V},\mathcal{E}), |  |

where each node vi∈𝒱v\_{i}\in\mathcal{V} represents a latent quantity (e.g., “number of adult lions”) and each directed edge (vj→vi)∈ℰ(v\_{j}\to v\_{i})\in\mathcal{E} encodes a functional dependency.
We restrict dependencies to elementary arithmetic operations:

|  |  |  |
| --- | --- | --- |
|  | vi=fi​((vj)j∈pa​(i)),fi∈{+,−,×,÷},v\_{i}=f\_{i}\bigl((v\_{j})\_{j\in\mathrm{pa}(i)}\bigr),\qquad f\_{i}\in\{+,-,\times,\div\}, |  |

where pa​(i)\mathrm{pa}(i) is the parent set of node ii.

Given numeric assignments to all leaf nodes, we define an evaluation map

|  |  |  |
| --- | --- | --- |
|  | val:𝒱→ℝ\mathrm{val}:\mathcal{V}\to\mathbb{R} |  |

recursively by

|  |  |  |
| --- | --- | --- |
|  | val​(vi)=fi​({val​(vj)}j∈pa​(i)),\mathrm{val}(v\_{i})=f\_{i}\bigl(\{\mathrm{val}(v\_{j})\}\_{j\in\mathrm{pa}(i)}\bigr), |  |

with base cases given by the leaf values.
For a designated query node v∗v^{\*}, the ground-truth answer is

|  |  |  |
| --- | --- | --- |
|  | a∗:=val​(v∗).a^{\*}:=\mathrm{val}(v^{\*}). |  |

In the GSM-Infinite implementation that we build upon [zhou2025gsminfinitellmsbehaveinfinitely], the query node v∗v^{\*} corresponds to:

* •

  the last numeric node in the topological order of the *forward* generator, or
* •

  the distinguished unknown parameter in the *equation-style reverse* generator.

Throughout, the DAG 𝒢\mathcal{G} is treated as the symbolic reasoning graph whose structure is shared across different numerical instantiations and linguistic realizations.

Reasoning Complexity.
We quantify the structural complexity of an instance by the number of arithmetic operations:

|  |  |  |
| --- | --- | --- |
|  | op​(𝒢)=|ℰ|.\mathrm{op}(\mathcal{G})=|\mathcal{E}|. |  |

This quantity lower-bounds the minimal length of the compositional reasoning chain needed to compute a∗a^{\*}, and is the primary knob we vary when studying extrapolative (depth-wise) generalization.

#### A.1.2 Abstract and Instance Parameters

Following the abstraction mechanism of GSM-Infinite, we explicitly separate *structure*, *numeric instantiation*, and *linguistic context*.

Abstract Parameters.
Each graph 𝒢\mathcal{G} is associated with a set of *abstract parameters* that:

* •

  specify which variables exist and how they decompose (e.g., that “total animals” decomposes into “lions” and “elephants”), and
* •

  determine the edge set ℰ\mathcal{E} and the operation fif\_{i} attached to each node.

These parameters define a purely symbolic graph, independent of particular numbers or entities.

Instance Parameters.
Given an abstract graph, *instance parameters* instantiate it with concrete values and entities:

* •

  numeric assignments to leaf nodes (e.g., “there are 12 adult lions and 7 elephant calves”), and
* •

  bindings of variables to context-specific surface forms (e.g., “adult lions in the city zoo”).

Instantiating different numeric values on the same abstract graph leads to a family of structurally identical problems that differ only in their concrete numbers.

Implicit Reasoning.
Not all abstract dependencies need to be explicitly verbalized in the natural-language problem.
For a given linguistic rendering, the edge set can be partitioned as

|  |  |  |
| --- | --- | --- |
|  | ℰ=ℰexplicit∪ℰimplicit,ℰexplicit∩ℰimplicit=∅,\mathcal{E}=\mathcal{E}\_{\mathrm{explicit}}\cup\mathcal{E}\_{\mathrm{implicit}},\qquad\mathcal{E}\_{\mathrm{explicit}}\cap\mathcal{E}\_{\mathrm{implicit}}=\emptyset, |  |

where (vj→vi)∈ℰexplicit(v\_{j}\to v\_{i})\in\mathcal{E}\_{\mathrm{explicit}} denotes a relation that is directly stated in the text (e.g., “there are 5 more elephants than lions”), while (vj→vi)∈ℰimplicit(v\_{j}\to v\_{i})\in\mathcal{E}\_{\mathrm{implicit}} denotes a relation that is part of the ground-truth reasoning graph but never directly verbalized (e.g., “total animals equals lions plus elephants”).
This separation allows explicit and implicit reasoning steps to coexist within the same underlying graph and enables us to probe models’ ability to recover unspoken dependencies.

#### A.1.3 Contextual Rendering

To map a symbolic graph to a natural-language problem, we introduce a contextual rendering function

|  |  |  |
| --- | --- | --- |
|  | Φ:(𝒢,τ)↦x,\Phi:(\mathcal{G},\tau)\mapsto x, |  |

where τ∈𝒯\tau\in\mathcal{T} is a *contextual template* and xx is the resulting text instance.

Templates.
A template τ\tau (e.g., animals–zoo, teachers–school, movie-festival) specifies:

* •

  how abstract variables are lexicalized into domain-specific surface forms (e.g., “adult lions”, “children in class A”, “tickets sold on day 1”), and
* •

  which subset of edges is realized explicitly in the wording, thereby determining the split between ℰexplicit\mathcal{E}\_{\mathrm{explicit}} and ℰimplicit\mathcal{E}\_{\mathrm{implicit}}.

For any two templates τa,τb∈𝒯\tau\_{a},\tau\_{b}\in\mathcal{T} that differ only in surface context, the induced problems remain structurally identical:

|  |  |  |
| --- | --- | --- |
|  | Struct​(Φ​(𝒢,τa))=Struct​(Φ​(𝒢,τb)),∀τa,τb∈𝒯,\mathrm{Struct}(\Phi(\mathcal{G},\tau\_{a}))=\mathrm{Struct}(\Phi(\mathcal{G},\tau\_{b})),\quad\forall\,\tau\_{a},\tau\_{b}\in\mathcal{T}, |  |

even though their surface realizations, entities, and explicit/implicit splits may differ.
Thus, a single abstract graph can be rendered into semantically distinct yet structurally equivalent problems, which we leverage to study contextual (breadth-wise) generalization.

Solution Format.
The rendering function produces a triple

|  |  |  |
| --- | --- | --- |
|  | x=([question],[solution],[answer]),x=\bigl(\text{[question]},\text{[solution]},\text{[answer]}\bigr), |  |

where:

* •

  [question] is the natural-language representation of the problem posed by the symbolic graph 𝒢\mathcal{G}, typically including a query regarding some aspect of the graph (e.g., ”How many tickets were sold on day 1?”). It abstracts away the underlying structure and provides the context for the solution.
* •

  [solution] is a step-by-step derivation that follows the topological order of the symbolic graph 𝒢\mathcal{G}. It includes intermediate reasoning steps and logical connections between the graph’s elements, ultimately leading to the final answer. The solution explicitly shows how each part of the problem is derived or calculated.
* •

  [answer] is the final response to the query posed in the [question], derived through the [solution] process. It is typically a numerical value or a specific entity that answers the question posed.

This structure ensures that the rendered output is both human-readable and logically consistent with the underlying symbolic graph, maintaining the integrity of the original problem while making it accessible in natural language.

Example Rendered Instance for teachers–school Context

[Question]

[question]
  
The number of elementary school in Westhaven City equals the number of public highschool in Westhaven City.
The number of elementary school in Evervale City equals the sum of the number of public highschool in Evervale City and the number of regional medical school in Westhaven City.
The total number of schools in Evervale City equals 22.
The number of elementary school in Brightford equals 3.
The number of public highschool in Brightford equals 2.
The number of regional medical school in Brightford equals the total number of schools in Westhaven City.
The number of regional medical school in Westhaven City equals 2.
The number of regional medical school in Evervale City equals 2 times the number of regional medical school in Brightford.
The number of public highschool in Westhaven City equals 3.
The number of public highschool in Evervale City exists, and its number is greater than 0.
How many public highschool does Evervale City have?
[/question]


Solution

[solution]
  
The question is difficult, so we use equations to solve it.
Define public highschool in Westhaven City as UU; so U=3U=3.
Define elementary school in Westhaven City as BB; so B=U=3B=U=3.
Define regional medical school in Westhaven City as hh; so h=2h=2.
Define total number of schools in Westhaven City as yy;
d=U+B=3+3=6d=U+B=3+3=6, so y=d+h=6+2=8y=d+h=6+2=8.
Define regional medical school in Brightford as QQ; so Q=y=8Q=y=8.
Define regional medical school in Evervale City as SS;
z=Q=8z=Q=8, so S=2​z=16S=2z=16.
Define public highschool in Evervale City as xx (unknown).
Define elementary school in Evervale City as mm; so m=x+h=x+2m=x+h=x+2.
Define total number of schools in Evervale City as kk.



n=x+(x+2)=2​x+2,k=n+S=2​x+18.n=x+(x+2)=2x+2,\qquad k=n+S=2x+18.
Since k=22k=22:



2​x+18=22,2​x=4,x=2.2x+18=22,\quad 2x=4,\quad x=2.
[/solution]


Answer

[answer] 2 [/answer]

#### A.1.4 Generation Pipeline and Structural Knobs

Our data generator follows a stage-wise procedure reminiscent of GSM-Infinite forward and reverse generators:

1. 1.

   Structural sampling.
   We first sample structural knobs that define the dependency graph:

   * •

     a target operation count range for op​(𝒢)\mathrm{op}(\mathcal{G});
   * •

     graph shape parameters (e.g., allowable in-degree, layering pattern) that control fan-in and depth; and
   * •

     operation types fi∈{+,−,×,÷}f\_{i}\in\{+,-,\times,\div\} attached to nodes.

   These choices determine a layered DAG 𝒢\mathcal{G} with a unique query node v∗v^{\*}.
2. 2.

   Abstract and instance parameterization.
   Given 𝒢\mathcal{G}, we sample abstract parameters (variable roles and decompositions) and instance parameters (numeric values on leaves) and evaluate all node values in topological order using the evaluation map val\mathrm{val} defined above.
3. 3.

   Contextual rendering.
   We choose a template τ∈𝒯\tau\in\mathcal{T} and apply the rendering function Φ​(𝒢,τ)\Phi(\mathcal{G},\tau) to obtain a natural-language triple (problem,question,solution)(\text{problem},\text{question},\text{solution}), deciding which dependencies are verbalized (explicit) and which remain implicit.
4. 4.

   Forward vs. reverse modes. Following [zhou2025gsminfinitellmsbehaveinfinitely], we support two modes of generation: In *forward* mode, we generate a standard arithmetic word problem where the final node in the topological order is queried.
   In *reverse* mode, we treat one node as an unknown and phrase an equation-style problem where the model must solve for that quantity, while the rest of the graph remains fully specified.

By jointly varying (i) the operation count op​(𝒢)\mathrm{op}(\mathcal{G}) and (ii) the template τ\tau, we obtain a clean two-dimensional testbed for studying depth scaling and context transfer.
The same framework is used to define distinct data distributions for pre-training, mid-training, and post-training by sampling from different regions of (op​(𝒢),τ)(\mathrm{op}(\mathcal{G}),\tau)-space.

#### A.1.5 Deduplication and Canonicalization

To guarantee cleanliness and avoid contamination across training and evaluation splits, we perform exact hash-based deduplication at the level of rendered triples.
Each instance is canonicalized by:

* •

  serializing the triple (problem,question,solution)(\text{problem},\text{question},\text{solution}) into a normalized string representation (e.g., stripping extraneous whitespace and normalizing numeric formatting), and
* •

  hashing this canonical form to obtain a global identifier.

We discard any duplicate hashes within and across splits, ensuring that no identical problem–solution triple appears in both training and evaluation.

### A.2 Task Setup

In real-world deployments, language models are expected to generalize reasoning along two complementary dimensions [setlur2025e3learningexploreenables, zhou2025doeslearningmathematicalproblemsolving, huan2025math].
Our controllable dataset makes these dimensions explicit and allows us to probe how *pre-training*, *mid-training*, and *post-training* shape each type of generalization.

Notation.
Let fθpref^{\text{pre}}\_{\theta}, fθmidf^{\text{mid}}\_{\theta}, and fθpostf^{\text{post}}\_{\theta} denote the language models after pre-training, after additional mid-training, and after post-training (RL), respectively.
We write Correct​(f,𝒢,τ)\mathrm{Correct}(f,\mathcal{G},\tau) for correctness on instances generated from graph 𝒢\mathcal{G} under template τ\tau, using the strict metric defined in the evaluation protocol below.

Extrapolative (Depth) Generalization.
We parameterize each training phase ϕ∈{pre,mid,post}\phi\in\{\text{pre},\text{mid},\text{post}\} by the range of operation counts it sees.
Let 𝒪ϕ\mathcal{O}\_{\phi} be the set of op​(𝒢)\mathrm{op}(\mathcal{G}) values present in the training distribution of phase ϕ\phi, and let

|  |  |  |
| --- | --- | --- |
|  | 𝒪train=𝒪pre∪𝒪mid∪𝒪post.\mathcal{O}\_{\text{train}}=\mathcal{O}\_{\text{pre}}\cup\mathcal{O}\_{\text{mid}}\cup\mathcal{O}\_{\text{post}}. |  |

An *in-distribution* evaluation condition uses graphs with op​(𝒢)∈𝒪train\mathrm{op}(\mathcal{G})\in\mathcal{O}\_{\text{train}}, while an *extrapolative* (out-of-distribution, OOD) condition evaluates on graphs with

|  |  |  |
| --- | --- | --- |
|  | op​(𝒢)>max⁡𝒪train.\mathrm{op}(\mathcal{G})>\max\mathcal{O}\_{\text{train}}. |  |

A model exhibits extrapolative generalization if it maintains high process-verified accuracy on these longer, unseen operations while remaining stable on in-distribution ones.
By the varied difficulty ranges populate 𝒪pre\mathcal{O}\_{\text{pre}}, 𝒪mid\mathcal{O}\_{\text{mid}}, and 𝒪post\mathcal{O}\_{\text{post}}, we can isolate how each phase contributes to depth-wise generalization.

Contextual (Breadth) Generalization.
A fixed reasoning graph 𝒢\mathcal{G} can be rendered into structurally equivalent instances under different templates,

|  |  |  |
| --- | --- | --- |
|  | Struct​(Φ​(𝒢,τa))=Struct​(Φ​(𝒢,τb))in principle,\mathrm{Struct}(\Phi(\mathcal{G},\tau\_{a}))=\mathrm{Struct}(\Phi(\mathcal{G},\tau\_{b}))\quad\text{in principle}, |  |

Our dataset is *randomly sampled* during training and does not deliberately align graphs across templates.
As a result, most graphs are observed only under a subset of contexts during training. Let 𝒯ϕtrain\mathcal{T}^{\text{train}}\_{\phi} denote the templates exposed during training phase ϕ\phi, and 𝒯eval\mathcal{T}^{\text{eval}} the broader evaluation pool, including long-tailed templates.
A model at phase ϕ\phi demonstrates contextual generalization if it preserves reasoning performance when the narrative surface form shifts, even when the new context was never encountered during training:

|  |  |  |
| --- | --- | --- |
|  | Acc​(fθϕ,𝒢,τa)≈Acc​(fθϕ,𝒢,τb),τb∉𝒯ϕtrain.\mathrm{Acc}(f^{\phi}\_{\theta},\mathcal{G},\tau\_{a})\approx\mathrm{Acc}(f^{\phi}\_{\theta},\mathcal{G},\tau\_{b}),\qquad\tau\_{b}\notin\mathcal{T}^{\text{train}}\_{\phi}. |  |

Under this setup, contextual generalization measures whether the model has learned transferable *reasoning primitives* rather than memorized task styles, allowing it to apply the same structural reasoning across known, unseen, and long-tailed narrative environments.

### A.3 Training Setup

#### A.3.1 Model Architecture

We conduct experiments using decoder-only Qwen2.5 Architecture [qwen2025qwen25technicalreport] models with 100M parameters. The detailed architecture configurations are as in Table [1](#A1.T1 "Table 1 ‣ A.3.1 Model Architecture ‣ A.3 Training Setup ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")

|  |  |
| --- | --- |
| Component | Configuration |
| Model Type | Qwen2.5 |
| Number of Layers | 12 |
| Hidden Size | 768 |
| Intermediate Size | 3,072 |
| Number of Attention Heads | 12 |
| Number of Key-Value Heads | 2 |
| Activation Function | SiLU |
| RMS Norm Epsilon | 1e-06 |

Table 1: Model architecture details for the 100M-parameter Qwen2.5 model used in experiments.

#### A.3.2 Tokenizer and Input Representation

We follow the *Physics of Language Models* series [ye2024physicslanguagemodels21] and train a byte-pair encoding (BPE) tokenizer directly on our synthetic reasoning corpus. The resulting vocabulary has 2,200 tokens (including special tokens). All problems, questions, and solutions are tokenized with a maximum sequence length of 2,048 tokens.

#### A.3.3 Hyperparameters

Pre-training.
All experiments start from a 100M-parameter Qwen2.5 model trained from scratch on our controllable reasoning corpus, using a 100×100\times token-to-parameter ratio, pre-training on 10B tokens.
We use a context length of 2048 tokens, batch-size 512K tokens, learning rate 2×10−42\times 10^{-4} with weight decay 0.10.1, cosine decay with minimum learning rate 3×10−53\times 10^{-5},
warmup ratio 5%5\%, and a single epoch over the corpus. All models are trained in bf16 precision.

Mid-training (Continue Pre-training).
Starting from the pre-trained checkpoint, we perform an additional and optional curriculum in § [5](#S5 "5 How Does Mid-Training Interact with Post-Training? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"). We train with maximum sequence length 2,048. We use a global batch size of 512K tokens, learning rate 1×10−41\times 10^{-4}, weight decay 0.10.1, cosine decay with minimum learning rate 3×10−53\times 10^{-5}, and a higher warmup ratio of 15%15\%.

Post-training.
Finally, we apply RL fine-tuning usin GRPO [shao2024deepseekmathpushinglimitsmathematical]. We use a global batch size of 1,024 examples, maximum prompt and response lengths of
1024 tokens, and two training epochs. The actor uses learning rate 1×10−61\times 10^{-6}, PPO mini-batch size 256, micro-batch size 16 per GPU, KL regularization with coefficient 10−310^{-3} (low-variance KL penalty), and zero
entropy bonus. During RL rollouts we sample with temperature TRL=1.0T\_{\text{RL}}=1.0, top-p=1.0p=1.0, and no top-kk truncation (full nucleus sampling). For offline evaluation and reporting we generate with temperature Teval=0.7T\_{\text{eval}}=0.7, top-p=1.0p=1.0, and top-k=−1k=-1 (no truncation), using a maximum of 1,024 new tokens per problem.

#### A.3.4 Performance Ladder

![Refer to caption](/html/2512.07783/assets/x9.png)

![Refer to caption](/html/2512.07783/assets/x10.png)

Figure 9: Pre-training dynamics across varying operation ranges: In-distribution tasks (op=2-10), edge-of-competence OOD tasks (op=11-14), and OOD-hard tasks (op=15-20). The plots show the performance measured by pass@k over training steps.

The performance ladder defines three key levels based on task difficulty: 1)In-distribution tasks (op=2-10): Aim for near-100% pass@128 accuracy; 2)OOD-edge tasks (op=11-14): Ensure non-zero pass@128 performance; 3) OOD-hards tasks (op=15-20): Aim for zero pass@128, signaling the model’s competence limits.
Post-training is performed on the edge of competence, ensuring the model generalizes to harder tasks. A breakdown of training dynamics across these performance levels is shown in Figure [9](#A1.F9 "Figure 9 ‣ A.3.4 Performance Ladder ‣ A.3 Training Setup ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

### A.4 Process-Verified Evaluation

Given an input instance with ground-truth graph (𝒢,a∗)(\mathcal{G},a^{\*}), the model produces a free-form solution ss.
We deterministically parse ss into a predicted dependency graph

|  |  |  |
| --- | --- | --- |
|  | 𝒢^=(𝒱^,ℰ^,val^),a^,\hat{\mathcal{G}}=(\hat{\mathcal{V}},\hat{\mathcal{E}},\widehat{\mathrm{val}}),\qquad\hat{a}, |  |

where nodes in 𝒱^\hat{\mathcal{V}} correspond to named intermediate quantities in the solution, ℰ^\hat{\mathcal{E}} encodes which previously defined quantities each step depends on, val^\widehat{\mathrm{val}} stores the inferred numeric value for each node, and a^\hat{a} is the extracted final answer.
The parser segments the solution into “Define …as …” steps, infers each step’s dependencies from the variables it uses, and evaluates the last computable arithmetic expression in the step (falling back to the last numeric literal if needed) to obtain a numeric value.
This yields a graph-level representation of the model’s reasoning trace aligned with the gold dependency graph.

Let the gold graph be

|  |  |  |
| --- | --- | --- |
|  | 𝒢=(𝒱,ℰ,val),a∗,\mathcal{G}=(\mathcal{V},\mathcal{E},\mathrm{val}),\qquad a^{\*}, |  |

with node set 𝒱\mathcal{V}, edge set ℰ\mathcal{E}, and value map val\mathrm{val}.
We evaluate the reasoning process at the *step level*.
For each gold node v∈𝒱v\in\mathcal{V}, define a per-step correctness indicator

|  |  |  |
| --- | --- | --- |
|  | s​(v;𝒢^,𝒢)={1,if ​v∈𝒱^,pa𝒢^​(v)=pa𝒢​(v),andval​(v),val^​(v)​ are both defined and ​val^​(v)=val​(v),0,otherwise,s(v;\hat{\mathcal{G}},\mathcal{G})=\begin{cases}1,&\text{if }v\in\hat{\mathcal{V}},\ \mathrm{pa}\_{\hat{\mathcal{G}}}(v)=\mathrm{pa}\_{\mathcal{G}}(v),\ \text{and}\\ &\quad\mathrm{val}(v),\widehat{\mathrm{val}}(v)\text{ are both defined and }\widehat{\mathrm{val}}(v)=\mathrm{val}(v),\\[4.0pt] 0,&\text{otherwise,}\end{cases} |  |

where pa𝒢​(v)\mathrm{pa}\_{\mathcal{G}}(v) and pa𝒢^​(v)\mathrm{pa}\_{\hat{\mathcal{G}}}(v) denote the parent sets (dependencies) of vv in the gold and predicted graphs, respectively.
Missing nodes, incorrect dependency sets, or mismatched values all yield s​(v;𝒢^,𝒢)=0s(v;\hat{\mathcal{G}},\mathcal{G})=0.

We then define the *process accuracy* of a predicted reasoning trace as the average step-level accuracy over all gold nodes:

|  |  |  |
| --- | --- | --- |
|  | ProcessAcc​(𝒢^;𝒢)=1|𝒱|​∑v∈𝒱s​(v;𝒢^,𝒢).\mathrm{ProcessAcc}(\hat{\mathcal{G}};\mathcal{G})=\frac{1}{|\mathcal{V}|}\sum\_{v\in\mathcal{V}}s(v;\hat{\mathcal{G}},\mathcal{G}). |  |

Extra predicted nodes v∈𝒱^∖𝒱v\in\hat{\mathcal{V}}\setminus\mathcal{V} are allowed and do not affect ProcessAcc\mathrm{ProcessAcc}; they correspond to redundant but compatible intermediate steps.

A prediction is regarded as fully correct only when both the reasoning graph and the final answer match.
We formalize this via a *verified correctness*:

|  |  |  |
| --- | --- | --- |
|  | VerifiedCorrect​(a^,𝒢^;a∗,𝒢)={1,if ​ProcessAcc​(𝒢^;𝒢)=1​and​a^=a∗,0,otherwise.\mathrm{VerifiedCorrect}(\hat{a},\hat{\mathcal{G}};\,a^{\*},\mathcal{G})=\begin{cases}1,&\text{if }\mathrm{ProcessAcc}(\hat{\mathcal{G}};\mathcal{G})=1\ \text{and}\ \hat{a}=a^{\*},\\[4.0pt] 0,&\text{otherwise.}\end{cases} |  |

Accordingly, all pass@k metrics (e.g., pass@1, pass@128) reported in this work treat a sample as correct only when the model (i) predicts every gold step correctly (step-level process accuracy =1=1) and (ii) produces the correct final answer.
This strict criterion ensures that reported gains reflect genuine, faithful reasoning rather than coincidental correctness.

### A.5 Training Dynamics for § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")

In this section, we provide a detailed analysis on the training dynamics for different post-training recipes in extrapolative generalization.
NLL Reduction Across Evaluation Ranges.
We analyze the post-training across different post-training data recipes used in § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") and their impact on NLL reduction across various evaluation operation ranges.

![Refer to caption](/html/2512.07783/assets/figures/composition_nll.png)


Figure 10: NLL reduction compared with the base model. White boxes denote RL-trained operation ranges. NLL gains decay smoothly as the evaluation range diverges from the RL-trained operations. Notably, RL on op=11-14 achieves the largest NLL reduction on op=15-20.

We can observe from Figure [10](#A1.F10 "Figure 10 ‣ A.5 Training Dynamics for § 3 ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") that post-training consistently reduces NLL across all evaluation ranges, with the most significant gains occurring in op=11-14 range. This indicates that the model effectively learns to compose atomic skills to tackle more complex problems.
Post-training Dynamics.
We further investigate the reward dynamics during post-training across different data recipes.

![Refer to caption](/html/2512.07783/assets/x11.png)


Figure 11: Reward dynamics across different post-training data recipes. RL on op=9-12 and op=11-14 tasks, which are calibrated to the model’s edge of competence, leads to genuine improvements in reasoning. However, when the task difficulty is either too easy or too hard, the reward stagnates, indicating limited learning progress.

From Figure [11](#A1.F11 "Figure 11 ‣ A.5 Training Dynamics for § 3 ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), we observe that post-training on tasks aligned with the model’s edge of competence (op=9-12 and op=11-14) leads to significant reward improvements, indicating effective learning. In contrast, when the tasks are too easy (op=7-10) or too hard (op=17-20), the reward plateaus, suggesting limited learning progress in these regimes.

### A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization

In this section, we provide a detailed analysis of how different post-training data recipes affect contextual generalization to long-tailed contexts given atomic reasoning primitives during pre-training.

#### A.6.1 When Reasoning Primitives are Shared During Pre-Training

Beyond mastering fundamental reasoning skills, an essential dimension of model generalization lies in contextual generalization—the capacity to transfer learned reasoning behaviors across diverse problem contexts, such as varying surface narratives or domains.
In this section, we investigate whether post-training can incentivize models to generalize reasoning competence to long-tailed or underrepresented contexts that were scarcely observed during pre-training.

Task Settting.
We study two distinct problem contexts: a frequent, canonical context A and a long-tailed context B, both sharing the same underlying reasoning priors (logical-arithmetic reasoning in our case, detailed context settings can be found in Appendix [A.9](#A1.SS9 "A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")).
The pre-training corpus consists of 99.9% context A and only 0.1% context B, both spanning op=2-20.
During post-training, we vary the exposure to context B across 200K samples with different ratios: 0%, 2%, 10%, 50%, and 100%.

![Refer to caption](/html/2512.07783/assets/x12.png)


Figure 12: pass@k performance on contextual generalization tasks after post-training with varying exposure to context B. With shared reasoning primitives during pre-training, models exhibit strong transfer to context B even with limited or no exposure during post-training.

Observation 5

With shared reasoning primitives during pre-training, there is a positive relation between exposure to context B during post-training and performance on context B. Notably, even without any context B exposure during post-training, the model still achieves significant transfer, underscoring the role of shared primitives in enabling contextual generalization.

Takeaway 5

When atomic primitives are shared, post-training can incentivize generalization to long-tailed contexts. Remarkably, even with a 0% exposure to context B during post-training, the model achieves substantial transfer, highlighting the critical role of shared reasoning structures during pre-training.

#### A.6.2 When Only Atomic Primitives are Exposed During Pre-Training

We next examine contextual generalization when the base model has only been exposed to basic atomic primitives in the long-tailed context during pre-training.

Task Setting.
With the same contextual data distribution as above, we restrict context B data during pre-training to only atomic operations, while context A spans the full range.
The pre-training corpus consists of 99% context A (op=2-20) and only 1% context B, with context B restricted to atomic operations (op=2).
Thus, the model learns reasoning structures primarily through context A, while having minimal exposure to the surface forms of context B.
During post-training, we perform RL fine-tuning with 200K samples where the ratio of context B data varies across five regimes: 0%, 1%, 10%, 50%, and 100%. Detailed data recipes can be found in Appendix [A.9](#A1.SS9 "A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

![Refer to caption](/html/2512.07783/assets/x13.png)


Figure 13: pass@k performance for different contexts with base model limited to basic atoms for context B. Post-training on context A maintains stable performance, while exposure of 10% context B during RL enables contextual transfer.

Observation 6

As shown in Figure [13](#A1.F13 "Figure 13 ‣ A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), post-training exclusively on context A or with only extremely sparse exposure to context B (0–1%) maintains strong performance within context A but yields minimal transfer to the long-tailed context B.
However, once a small amount of context B data is introduced—around 10% of total samples—context B performance improves dramatically, with pass@128 accuracy increasing by over +76+76 points.
Further increasing the proportion of context B data (50%, 100%) brings diminishing gains, indicating that RL rapidly establishes robust cross-context reasoning once minimal supervision is available. Notably, even when post-training uses 100% context B data—entirely distinct from the dominant pre-training context—context A performance remains stable.
This shows that RL enables model to learn transferable reasoning policies that extend across surface forms while preserving competence in previously mastered contexts.

Takeaway 6

RL enables stable cross-context generalization under extreme imbalance.
Even when the base model has only minimal exposure to long-tailed contexts during pre-training, RL fine-tuning can transfer reasoning competence across domains by leveraging shared reasoning structures.

#### A.6.3 Training Dynamics for § [A.6.2](#A1.SS6.SSS2 "A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")

We plot the post-training reward dynamics across different data recipes used in § [A.6.2](#A1.SS6.SSS2 "A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") to further understand how varying exposure to long-tailed contexts during RL affects learning progress.

![Refer to caption](/html/2512.07783/assets/x14.png)


Figure 14: Reward dynamics across different post-training data recipes. When RL exposure to context B is extremely limited (0-1%), the reward stagnates. However, with moderate exposure (10-100%), the reward improves significantly, reflecting effective learning and transfer.

From Figure [14](#A1.F14 "Figure 14 ‣ A.6.3 Training Dynamics for § A.6.2 ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), we can observe that when the exposure to context B during post-training is extremely limited (0-1%), the reward plateaus, indicating minimal learning progress. However, with moderate exposure (10-100%), the reward improves significantly, reflecting effective learning and transfer to the long-tailed context.

### A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization

Pre-training defines the atomic reasoning primitives that post-training can later compose and extend.
If the base model already encounters moderately complex problems during pre-training, post-training may push those primitives toward deeper, compositional reasoning.
Otherwise, post-training may lack the scaffolding to explore beyond its inherited competence. We thus study how varying pre-training difficulty influences subsequent extrapolative generalization.

Task Setting.
We fix the post-training recipe to 200K samples from the op=11-14 range, previously identified as a edge of competence (see Figure [3](#S3.F3 "Figure 3 ‣ 3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")). We then vary the proportion of “hard” data (op=7-10) included during pre-training to assess how exposure to complex primitives affects the base model’s ability to generalize after RL.
(See Appendix [A.9](#A1.SS9 "A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") for detailed data recipes.)

![Refer to caption](/html/2512.07783/assets/x15.png)


Figure 15: pass@128 performance on extrapolative tasks after post-training on op=11-14, under varying levels of hard-data exposure during pre-training.

Observation 7

As shown in Figure [15](#A1.F15 "Figure 15 ‣ A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), greater exposure to hard problems during pre-training consistently improves both base and post-trained performance.
However, the marginal gain from RL diminishes as pre-training becomes more comprehensive. When pre-training already covers a substantial fraction of mid-depth tasks, RL adds only modest improvement.
By contrast, when pre-training includes limited but nontrivial exposure to difficult primitives (e.g., 20% of op=7-10), RL produces the largest relative boost—enhancing pass@128 accuracy on op=15-20 by more than +22+22 points.
This suggests that RL is most effective when the model’s prior competence is partial—strong enough to support exploration, but incomplete enough to leave room for discovery.

Takeaway 7

Pre-training establishes the foundation, RL extends it.
Rich exposure to compositional primitives during pre-training enables RL to push reasoning depth beyond the pre-training range. Yet the benefits of RL taper off once those primitives are fully mastered, highlighting the complementary roles of the two stages.

#### A.7.1 Training Dynamics for § [A.7](#A1.SS7 "A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")

We analyze the training dynamics during post-training across different pre-training data recipes.

![Refer to caption](/html/2512.07783/assets/x16.png)


Figure 16: Reward dynamics across different pre-training data recipes. Models with moderate hard-data exposure (20-50%) during pre-training exhibit significant reward improvements during post-training, indicating effective learning and extrapolation. In contrast, models with either too little (0%) or too much (100%) hard-data exposure show limited reward gains, suggesting constrained learning progress.

### A.8 Training Dynamics for § [4](#S4 "4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models")

In this section, we provide an analysis of the training dynamics for different pre-training data recipes in contextual generalization in § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").

![Refer to caption](/html/2512.07783/assets/x17.png)


Figure 17: Reward dynamics across different pre-training data recipes.
Models with minimal exposure to long-tailed contexts exhibit no reward improvement during post-training. While models with moderate to full exposure show significant reward improvements, indicating effective learning and contextual generalization.

From Figure [17](#A1.F17 "Figure 17 ‣ A.8 Training Dynamics for § 4 ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), we observe that moderate exposure ratio to long-tailed contexts, even with basic primitives during pre-training, is necessary for the model to make significant reward improvements during post-training.

### A.9 Post-Training and Pre-Training Data Recipe

In this section, we detail the data recipes employed in § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") § [4](#S4 "4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), § [A.6.1](#A1.SS6.SSS1 "A.6.1 When Reasoning Primitives are Shared During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), § [A.6.2](#A1.SS6.SSS2 "A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), and § [A.7](#A1.SS7 "A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models").
Table [2](#A1.T2 "Table 2 ‣ A.9 Post-Training and Pre-Training Data Recipe ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") summarizes the specific operation count ranges, contextual templates, and training budgets utilized across different experimental sections.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Pre-training | | | Post-training (RL) | | |
| Section | op​(𝒢)\mathrm{op}(\mathcal{G}) | Contexts | Training Budget | op​(𝒢)\mathrm{op}(\mathcal{G}) | Contexts | Training Budget |
| § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") | 20%​op=2-4+30%​op=5-7+50%​op=8-1020\%\texttt{op=2-4}+30\%\texttt{op=5-7}+50\%\texttt{op=8-10} | 33%A+33%B+33%C | 10B tokens | \cellcolorgray!15op=8-10 | 33%A+33%B+33%C | 204.8k samples |
| \cellcolorgray!15op=9-12 |
| \cellcolorgray!15op=11-14 |
| \cellcolorgray!15op=17-20 |
| § [4](#S4 "4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") | \cellcolorgray!15100%​op=2-20​A+0%​op=2​B100\%\texttt{op=2-20}\,\mathrm{A}+0\%\texttt{op=2}\,\mathrm{B} | | 10B tokens | op=2-20 | 50%50\% A + 50%50\% B | 204.8k samples |
| \cellcolorgray!1599.9%​op=2-20​A+0.1%​op=2​B99.9\%\texttt{op=2-20}\,\mathrm{A}+0.1\%\texttt{op=2}\,\mathrm{B} | |
| \cellcolorgray!1599%​op=2-20​A+1%​op=2​B99\%\texttt{op=2-20}\,\mathrm{A}+1\%\texttt{op=2}\,\mathrm{B} | |
| \cellcolorgray!1590%​op=2-20​A+10%​op=2​B90\%\texttt{op=2-20}\,\mathrm{A}+10\%\texttt{op=2}\,\mathrm{B} | |
| § [A.6.1](#A1.SS6.SSS1 "A.6.1 When Reasoning Primitives are Shared During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") | op=2-20 | 99.9%A+0.1%B | 10B tokens | op=2-20 | \cellcolorgray!15100%100\% A | 204.8k samples |
| \cellcolorgray!1598%98\%A + 2%2\%B |
| \cellcolorgray!1590%90\%A + 10%10\%B |
| \cellcolorgray!1550%50\%A + 50%50\%B |
|  |  | \cellcolorgray!15100%100\%B |
| § [A.6.2](#A1.SS6.SSS2 "A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") | 99%​op=2-20​A+1%​op=2​B99\%\texttt{op=2-20}\,\mathrm{A}+1\%\texttt{op=2}\,\mathrm{B} | | 10B tokens | op=2-20 | \cellcolorgray!15100%100\% A | 204.8k samples |
| \cellcolorgray!1599%99\%A + 1%1\%B |
| \cellcolorgray!1590%90\%A + 10%10\%B |
| \cellcolorgray!1550%50\%A + 50%50\%B |
| \cellcolorgray!15100%100\%B |
| § [A.7](#A1.SS7 "A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") | \cellcolorgray!1599.9%​op=2-6+0.1%​op=8-2099.9\%\,\texttt{op=2-6}+0.1\%\,\texttt{op=8-20} | 33%A+33%B+33%C | 10B tokens | op=11-14 | 33%A+33%B+33%C | 204.8k samples |
| \cellcolorgray!1549.95%​op=2-4+49.95%​op=5-7+0.1%​op=8-1049.95\%\,\texttt{op=2-4}+49.95\%\,\texttt{op=5-7}+0.1\%\,\texttt{op=8-10} |
| \cellcolorgray!1547.5%​op=2-4+47.5%​op=5-7+5%​op=8-1047.5\%\,\texttt{op=2-4}+47.5\%\,\texttt{op=5-7}+5\%\,\texttt{op=8-10} |
| \cellcolorgray!1550%​op=2-4+30%​op=5-7+20%​op=8-1050\%\,\texttt{op=2-4}+30\%\,\texttt{op=5-7}+20\%\,\texttt{op=8-10} |
| \cellcolorgray!1520%​op=2-4+30%​op=5-7+50%​op=8-1020\%\,\texttt{op=2-4}+30\%\,\texttt{op=5-7}+50\%\,\texttt{op=8-10} |

Table 2: Data recipes for pre-/post-training experiments in § [3](#S3 "3 When Does Post-Training Incentivize Reasoning Beyond the Base Model? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), § [4](#S4 "4 How Does Pre-training Exposure Shape Post-Training Generalization? ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), § [A.6.1](#A1.SS6.SSS1 "A.6.1 When Reasoning Primitives are Shared During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), § [A.6.2](#A1.SS6.SSS2 "A.6.2 When Only Atomic Primitives are Exposed During Pre-Training ‣ A.6 Detailed Analysis of Post-Training Effects on Contextual Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), and § [A.7](#A1.SS7 "A.7 Detailed Analysis of Pre-Training Effects on Extrapolative Generalization ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"). op​(𝒢)\mathrm{op}(\mathcal{G}) ranges indicate the operation counts during each training phase.
Contexts A, B, C correspond to distinct templates: A = animals–zoo, B = teachers–school, C = movie-festival.
The data recipes for different operation ranges and contexts are uniformly sampled within the specified proportions.
Shaded cells indicate the ablated settings.

### A.10 Mid-/Post-Training Mixing with Different Computation Budget

In this section, we first detail the compute budget formulation for mid-training and RL equivalence, then provide the exact data recipes for combining mid-training and post-training under different total compute budgets.

#### A.10.1 Compute Budget of Mid-Training and RL Equivalence

Training Computation.
Following the Chinchilla scaling law [hoffmann2022trainingcomputeoptimallargelanguage], a decoder-only Transformer with PP non-embedding parameters trained on TT tokens consumes approximately

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ctrain≈6​P​Tf​l​o​p​s.C\_{\text{train}}\approx 6P\,T\quad flops. |  | (4) |

Thus, a mid-training phase with budget TmidT\_{\text{mid}} incurs Cmid=6​P​Tmidf​l​o​p​sC\_{\text{mid}}=6P\,T\_{\text{mid}}\quad flops.

Fine-Grained RL Computation.
For on-policy GRPO, computation can be decomposed as:

* •

  Rollout: actor model forward (2​P2P),
* •

  Reference (optional): reference model forward (2​P2P),
* •

  Policy Update: forward (2​P2P) and backward (4​P4P) passes.

Summing these terms yields:

|  |  |  |  |
| --- | --- | --- | --- |
|  | CRL=(8+2​γ)​P​N​r​Ltotal,C\_{\mathrm{RL}}=(8+2\gamma)P\,N\,r\,L\_{\text{total}}, |  | (5) |

where γ∈{0,1}\gamma\in\{0,1\} toggles the reference-model pass, NN is the number of RL samples, rr is the rollout size, and LtotalL\_{\text{total}} is the total sequence length (including both prompt and completion).

Mid-training Token Equivalence.
Normalizing by Equation [4](#A1.E4 "Equation 4 ‣ A.10.1 Compute Budget of Mid-Training and RL Equivalence ‣ A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") gives the equivalent mid-training token cost:

|  |  |  |  |
| --- | --- | --- | --- |
|  | TRL=CRL6​P=(43+γ3)​N​r​Ltotal.T\_{\mathrm{RL}}=\frac{C\_{\mathrm{RL}}}{6P}=\Bigl(\tfrac{4}{3}+\tfrac{\gamma}{3}\Bigr)NrL\_{\text{total}}. |  | (6) |

When γ=1\gamma=1, we obtain the equivalence used in the main text:

|  |  |  |
| --- | --- | --- |
|  | TRL=53NrLtotal.\boxed{T\_{\mathrm{RL}}=\tfrac{5}{3}NrL\_{\text{total}}.} |  |

Budget Allocation and Step Calculation.
Given total budget TT and RL ratio β\beta,

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Tmid\displaystyle T\_{\text{mid}} | =(1−β)⋅T,\displaystyle=(1-\beta)\cdot T, | TRL,eq\displaystyle T\_{\text{RL,eq}} | =β⋅T.\displaystyle=\beta\cdot T. |  | (7) |

The corresponding number of RL samples N​(p)N(p) and update steps are:

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | N​(β)\displaystyle N(\beta) | =35⋅β​Tr​Ltotal,\displaystyle=\frac{3}{5}\cdot\frac{\beta T}{rL\_{\text{total}}}, | stepsRL​(p)\displaystyle\text{steps}\_{\text{RL}}(p) | =N​(β)B,\displaystyle=\frac{N(\beta)}{B}, |  | (8) |

where r=6r=6 is the rollout size, Ltotal=2048L\_{\text{total}}=2048 is the total sequence length, B=1024B=1024 is the RL batch size, and TT is the total token budget.
The mid-training steps are:

|  |  |  |  |
| --- | --- | --- | --- |
|  | stepsmid​(β)=TmidBmid⋅Lmid,\text{steps}\_{\text{mid}}(\beta)=\frac{T\_{\text{mid}}}{B\_{\text{mid}}\cdot L\_{\text{mid}}}, |  | (9) |

where Bmid=512×1024B\_{\text{mid}}=512\times 1024 is the mid-training batch size and Lmid=2048L\_{\text{mid}}=2048 is the mid-training sequence length.

| Total | Mid-Only | RL-Only | | 80% Mid / 20% RL | | 50% Mid / 50% RL | | 20% Mid / 80% RL | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (B) | stepsmid\text{steps}\_{\text{mid}} | stepsRL\text{steps}\_{\text{RL}} | Samp.(k) | stepsmid\text{steps}\_{\text{mid}} | stepsRL\text{steps}\_{\text{RL}} | stepsmid\text{steps}\_{\text{mid}} | stepsRL\text{steps}\_{\text{RL}} | stepsmid\text{steps}\_{\text{mid}} | stepsRL\text{steps}\_{\text{RL}} |
| 1.05 | 2,000 | 50 | 51.2 | 1,600 | 10 | 1,000 | 25 | 400 | 40 |
| 2.10 | 4,000 | 100 | 102.4 | 3,200 | 20 | 2,000 | 50 | 800 | 80 |
| 4.20 | 8,000 | 200 | 204.8 | 6,400 | 40 | 4,000 | 100 | 1,600 | 160 |
| 8.40 | 16,000 | 400 | 409.6 | 12,800 | 80 | 8,000 | 200 | 3,200 | 320 |
| 12.58 | 24,000 | 600 | 614.4 | 19,200 | 120 | 12,000 | 300 | 4,800 | 480 |
| 16.78 | 32,000 | 800 | 819.2 | 25,600 | 160 | 16,000 | 400 | 6,400 | 640 |
| 20.00 | 38,147 | 954 | 976.6 | 30,517 | 191 | 19,073 | 477 | 7,629 | 763 |

Table 3: Experimental configurations across varying compute budget scales. We fix the mid-training batch size at 512K tokens. The table maps the total token budget TT to the specific step counts required for pure mid-training (p=1.0p=1.0), pure RL (p=0.0p=0.0), and hybrid splits.

Task Setting.
We use 10B tokens with 20% op=2-4, 30% op=5-7, and 50% op=8-10 for pre-training. To avoid catastrophic forgetting during mid-training, we use 20% budget for op=2-10 and 80% for op=11-14 during mid-training. For fair comparison, RL is performed with the same data distribution as mid-training.
Table [3](#A1.T3 "Table 3 ‣ A.10.1 Compute Budget of Mid-Training and RL Equivalence ‣ A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models") details the exact step counts for mid-training and RL across varying total token budgets TT and mid-training ratios pp.
We perform mid-/post-training with Full mid-training, Full RL, Light-RL (β=0.2\beta=0.2), Medium-RL (β=0.5\beta=0.5), and Heavy-RL (β=0.8\beta=0.8) under different total compute budgets.

![Refer to caption](/html/2512.07783/assets/x18.png)


Figure 18: pass@k performance for different mid-training and RL mixing ratios under varying total compute budgets.

Observation 8

As shown in Figure [18](#A1.F18 "Figure 18 ‣ A.10.1 Compute Budget of Mid-Training and RL Equivalence ‣ A.10 Mid-/Post-Training Mixing with Different Computation Budget ‣ Appendix A Appendix ‣ On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"), across all compute budgets Light-RL achieves the best OOD-edge pass@1. While Heavy-RL consistently attains the highest OOD-hard pass@1 performance.
For pass@128, when the compute budget is limited (4.2B tokens), Heavy-RL achieves the best performance in the OOD-hard setting. When the budget increases (8.4B tokens and above), Full RL attains the highest OOD-hard pass@128 performance.

Takeaway 8

Mid-training and post-training complement each other across varying compute budgets.
A combination of mid-training and RL post-training consistently outperforms either approach individually for pass@1 tasks.
For pass@128, the optimal post-training allocation depends on the available compute budget: with limited resources, allocating around 80% to RL strikes a balance between stability and exploration, while with more compute, full RL maximizes extrapolative gains.

[◄](/html/2512.07782)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2512.07783)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2512.07783)
[View original  
on arXiv](https://arxiv.org/abs/2512.07783)[►](/html/2512.07784)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Jan 6 05:54:34 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
